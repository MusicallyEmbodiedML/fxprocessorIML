#include "src/memllib/audio/AudioDriver.hpp"
#include "src/memllib/hardware/memlnaut/MEMLNaut.hpp"
#include <memory>
#include "src/memllib/interface/MIDIInOut.hpp"
#include "src/memllib/PicoDefs.hpp"
#include "src/memllib/interface/UARTInput.hpp"

// Example apps and interfaces
#include "src/memllib/examples/IMLInterface.hpp"

#include "src/memllib/hardware/memlnaut/Display.hpp"


/**
 * @brief FX processor audio app
 *
 */

#include <cmath>
#include "src/memllib/synth/maximilian.h"
#include "src/memllib/audio/AudioAppBase.hpp"
#include "src/memllib/synth/OnePoleSmoother.hpp"

volatile float input_level = 0;
volatile float input_pitch = 0;

class FXProcessorAudioApp : public AudioAppBase
{
public:
    AudioDriver::codec_config_t GetDriverConfig() const override {
        return {
            .mic_input = true,
            .line_level = 3,
            .mic_gain_dB = 20,
            .output_volume = 0.8f
        };
    }

    struct Params {
        float f_drift;
        float env_release;
        float dns_ratio;
    };
    static constexpr size_t kN_Params = sizeof(Params) / sizeof(float);

    const struct {
        float min = 40.f;
        float max = 400.f;
    } freq_range;

    FXProcessorAudioApp() : AudioAppBase(),
        setup_(false),
        smoother_(0.001, kSampleRate),
        target_params_(kN_Params, 0),
        smoothed_params_(kN_Params, 0) {}

    void Setup(float sample_rate, std::shared_ptr<InterfaceBase> interface) override
    {
        AudioAppBase::Setup(sample_rate, interface);
        // Additional setup code specific to FMSynthAudioApp
        // Set param smoothers
        smoother_.SetTimeMs(50.f);

        // Pitch detector setup
        bandpass_low.set(maxiBiquad::LOWPASS,
                         freq_range.max,
                         0.707f, 0);
        bandpass_high.set(maxiBiquad::HIGHPASS,
                          freq_range.min,
                          0.707f, 0);
        pitch_detector.setup();

        // Synth setup
        env.setAttack(10);
        env.setRelease(300);
        for (size_t n = 0; n < freq_scalings_.size(); n++) {
            osc_[n].UpdateParams();
        }

        // Downsampler
        dns_lpf_.set(maxiBiquad::LOWPASS,
                     250.f,
                     5.f, 0);

        // Default parameters
        Params default_params {
            .f_drift = 0.97,
            .env_release = 100,
            .dns_ratio = 15
        };
        params_ = default_params;

        // Setup finished
        setup_ = true;
    }

    stereosample_t Process(const stereosample_t x) override
    {
        WRITE_VOLATILE(input_level, std::abs(x.L) + std::abs(x.R));
        if (!setup_) {
            return { 0, 0 };
        }
        // Smooth parameters
        SmoothParams_();

        float dry = x.L;
        env.play(dry);
        // Pitch detection
        float detect = bandpass_low.play(dry);
        detect = bandpass_high.play(detect);
        float pitch = pitch_detector.process(detect);
        WRITE_VOLATILE(input_pitch, pitch);
        // if (pitch < freq_range.min || pitch > freq_range.max) {
        //     pitch = 0;
        // }

        // Synth
        float synth = 0;
        std::array<float, 2> scalings {
            params_.f_drift,
            1.f/params_.f_drift
        };
        for (size_t n = 0; n < freq_scalings_.size(); n++) {
            synth += osc_[n].sawn(pitch * scalings[n])
                     * (1.f/freq_scalings_.size());
        }

        // Decimation
        float decim = dns_.play(dry, kSampleRate / params_.dns_ratio);
        decim *= 3.f;
        //decim = std::tanh(decim);
        //decim = quant_.play(decim, 16);

        // Output
        float out_env = env.getEnv();
        out_env *= 6.f;
        out_env = std::pow(out_env, 1.8f);
        out_env = std::tanh(out_env);
        // if (out_env < 0.01) {
        //     out_env = 0;
        // }
        float yL = synth * out_env;
        float yR = yL; //decim;

        return { yL, yR };
    }

    void ProcessParams(const std::vector<float>& params) override
    {
        target_params_ = params;
    }

protected:

    bool setup_;

    // Smooth parameters
    std::vector<float> target_params_;
    std::vector<float> smoothed_params_;
    OnePoleSmoother<kN_Params> smoother_;
    Params params_;

    // Pitch detector:
    // - Bandpass biquad
    maxiBiquad bandpass_low;
    maxiBiquad bandpass_high;
    // - Zero crossing detector
    maxiZeroCrossingAvg pitch_detector;

    // Synth:
    // - Envelope
    maxiEnvelopeFollowerF env;
    // - Oscillator
    static constexpr std::array<float, 2> freq_scalings_
            {0.97f, 1.03f};
    std::array<maxiOsc, freq_scalings_.size()> osc_;

    // Downsampler
    // - downsample
    maxiDownSample dns_;
    // - decimate
    maxiBitQuant quant_;
    maxiBiquad dns_lpf_;

    /**
     * @brief Linear mapping function
     *
     * @param x float between 0 and 1
     * @param out_min minimum output value
     * @param out_max maximum output value
     * @return float Interpolated value between out_min and out_max
     */
    static __attribute__((always_inline)) float LinearMap_(float x, float out_min, float out_max)
    {
        return out_min + (x * (out_max - out_min));
    }

    /**
     * @brief S-curve mapping function
     *
     * @param x float between 0 and 1
     * @param out_min minimum output value
     * @param out_max maximum output value
     * @param curve_slope slope of the curve, 0 is linear, 1 is steep
     */
    static __attribute__((always_inline)) float SCurveMap_(float x, float out_min, float out_max, float curve_slope) {
        // Fast clamp using branchless min/max
        x = x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
        curve_slope = curve_slope < 0.0f ? 0.0f : (curve_slope > 1.0f ? 1.0f : curve_slope);

        // Pre-compute constants and reuse values
        const float centered = x - 0.5f;
        const float slope_factor = curve_slope * 14.0f + 1.0f;
        const float curved = centered * slope_factor;

        // Fast exp approximation for sigmoid (4th order minimax approximation)
        // Only valid for input range [-5, 5], which is fine for our use case
        float exp_x = -curved;
        const float x2 = exp_x * exp_x;
        exp_x = 1.0f + exp_x + (x2 * 0.5f) + (x2 * exp_x * 0.166666667f) + (x2 * x2 * 0.041666667f);
        const float sigmoid = 1.0f / exp_x;

        // Optimized linear interpolation
        const float range = out_max - out_min;
        const float result = ((1.0f - curve_slope) * x + curve_slope * sigmoid);
        return fma(result, range, out_min);
    }

    void SmoothParams_() {
        smoother_.Process(target_params_.data(), smoothed_params_.data());

        auto param_ptr = smoothed_params_.data();
        // Assign smoothed parameters to their functions
        params_.f_drift = LinearMap_(*param_ptr++, 0.86, 0.995);
        params_.dns_ratio = SCurveMap_(*param_ptr++, 3, 18, 0.5);
        params_.env_release = LinearMap_(*param_ptr++, 10, 500);
        env.setRelease(params_.env_release);
    }
};


/******************************* */


// Global objects
using CURRENT_AUDIO_APP = FXProcessorAudioApp;
using CURRENT_INTERFACE = IMLInterface;
std::shared_ptr<CURRENT_INTERFACE> interface;
std::shared_ptr<CURRENT_AUDIO_APP> audio_app;
std::shared_ptr<MIDIInOut> midi_interf;
std::shared_ptr<UARTInput> uart_input;
std::shared_ptr<Display> display;

// Inter-core communication
volatile bool core_0_ready = false;
volatile bool core_1_ready = false;
volatile bool serial_ready = false;
volatile bool interface_ready = false;

// We're only bound to the joystick inputs (x, y, rotate)
const size_t kN_InputParams = 2;
const std::vector<size_t> kUARTListenInputs {};


void bind_interface(std::shared_ptr<CURRENT_INTERFACE> &interface)
{
/*
    // Set up momentary switch callbacks
    MEMLNaut::Instance()->setMomA1Callback([interface] () {
        interface->Randomise();
        if (display) {
            display->post("Randomised");
        }
    });
    MEMLNaut::Instance()->setMomA2Callback([interface] () {
        interface->ClearData();
        if (display) {
            display->post("Dataset cleared");
        }
    });
*/
    MEMLNaut::Instance()->setMomB2Callback([interface] () {
        Serial.println("MOM_B2 pressed");
    });
    MEMLNaut::Instance()->setMomB1Callback([interface] () {
        Serial.println("MOM_B1 pressed");
    });
    MEMLNaut::Instance()->setMomA2Callback([interface] () {
        Serial.println("MOM_A2 pressed");
    });

    // Set up toggle switch callbacks
/*
    MEMLNaut::Instance()->setTogA1Callback([interface] (bool state) {
        if (display) {
            display->post(state ? "Training mode" : "Inference mode");
        }
        interface->SetTrainingMode(state ? CURRENT_INTERFACE::TRAINING_MODE : CURRENT_INTERFACE::INFERENCE_MODE);
        if (display && state == false) {
            display->post("Model trained");
        }
    });
*/
    MEMLNaut::Instance()->setTogA1Callback([interface] (bool state) {
        if (state) {
            Serial.println("TOG_A1 pressed");
        }
    });
    MEMLNaut::Instance()->setTogA2Callback([interface] (bool state) {
        if (state) {
            Serial.println("TOG_A2 pressed");
        }
    });
    MEMLNaut::Instance()->setTogB1Callback([interface] (bool state) {
        if (state) {
            Serial.println("TOG_B1 pressed");
        }
    });
    MEMLNaut::Instance()->setTogB2Callback([interface] (bool state) {
        if (state) {
            Serial.println("TOG_B2 pressed");
            // Randomise
            interface->SetTrainingMode(CURRENT_INTERFACE::TRAINING_MODE);
            interface->Randomise();
        }
    });

    MEMLNaut::Instance()->setJoySWCallback([interface] (bool state) {
        interface->SaveInput(state ? CURRENT_INTERFACE::STORE_VALUE_MODE : CURRENT_INTERFACE::STORE_POSITION_MODE);
        if (display) {
            display->post(state ? "Where do you want it?" : "Here!");
        }
    });

    // Set up joystick callbacks
    if (kN_InputParams > 0) {
        MEMLNaut::Instance()->setJoyXCallback([interface] (float value) {
            interface->SetInput(0, value);
        });
        MEMLNaut::Instance()->setJoyYCallback([interface] (float value) {
            interface->SetInput(1, value);
        });
/*
        MEMLNaut::Instance()->setJoyZCallback([interface] (float value) {
            interface->SetInput(2, value);
        });
*/
    }
/*
    // Set up other ADC callbacks
    MEMLNaut::Instance()->setRVZ1Callback([interface] (float value) {
        // Scale value from 0-1 range to 1-3000
        value = 1.0f + (value * 2999.0f);
        interface->SetIterations(static_cast<size_t>(value));
    });
*/

    // Set up loop callback
    MEMLNaut::Instance()->setLoopCallback([interface] () {
        interface->ProcessInput();
    });

    MEMLNaut::Instance()->setRVGain1Callback([interface] (float value) {
        //AudioDriver::setDACVolume(value);
        //Serial.println(value*4);
        //Serial.println("ADCDAC bypassed!");
        AudioDriver::setDACVolume(3.9f);
    });
}

void bind_uart_in(std::shared_ptr<CURRENT_INTERFACE> &interface) {
    if (uart_input) {
        uart_input->SetCallback([interface] (const std::vector<float>& values) {
            for (size_t i = 0; i < values.size(); ++i) {
                interface->SetInput(kN_InputParams + i, values[i]);
            }
        });
    }
}

void bind_midi(std::shared_ptr<CURRENT_INTERFACE> &interface) {
    if (midi_interf) {
        midi_interf->SetCCCallback([interface] (uint8_t cc_number, uint8_t cc_value) {
            Serial.printf("MIDI CC %d: %d\n", cc_number, cc_value);
        });
    }
}


void setup()
{
    Serial.begin(115200);
    //while (!Serial) {}
    Serial.println("Serial initialised.");
    WRITE_VOLATILE(serial_ready, true);

    // Setup board
    MEMLNaut::Initialize();
    pinMode(33, OUTPUT);
    display = std::make_shared<Display>();
    display->setup();
    display->post("MEML FX Unit");

    // Move MIDI setup after Serial is confirmed ready
    Serial.println("Initializing MIDI...");
    midi_interf = std::make_shared<MIDIInOut>();
    midi_interf->Setup(CURRENT_AUDIO_APP::kN_Params);
    midi_interf->SetMIDISendChannel(1);
    Serial.println("MIDI setup complete.");

    delay(100); // Allow Serial2 to stabilize

    // Setup UART input
    uart_input = std::make_shared<UARTInput>(kUARTListenInputs);
    const size_t total_input_params = kN_InputParams + kUARTListenInputs.size();

    // Setup interface with memory barrier protection
    {
        auto temp_interface = std::make_shared<CURRENT_INTERFACE>();
        MEMORY_BARRIER();
        temp_interface->setup(total_input_params, CURRENT_AUDIO_APP::kN_Params);
        MEMORY_BARRIER();
        temp_interface->SetMIDIInterface(midi_interf);
        MEMORY_BARRIER();
        interface = temp_interface;
        MEMORY_BARRIER();
    }
    WRITE_VOLATILE(interface_ready, true);

    // Bind interface after ensuring it's fully initialized
    bind_interface(interface);
    Serial.println("Bound interface to MEMLNaut.");
    bind_uart_in(interface);
    Serial.println("Bound interface to UART input.");
    bind_midi(interface);
    Serial.println("Bound interface to MIDI input.");

    WRITE_VOLATILE(core_0_ready, true);
    while (!READ_VOLATILE(core_1_ready)) {
        MEMORY_BARRIER();
        delay(1);
    }

    Serial.println("Finished initialising core 0.");
}

void loop()
{
    static uint32_t last_1ms = 0;
    static uint32_t last_10ms = 0;
    uint32_t current_time = micros();

    // Tasks to run as fast as possible
    {
        // Poll the UART input
        uart_input->Poll();
        // Poll the MIDI interface
        midi_interf->Poll();
    }

    // Tasks to run every 1ms
    if (current_time - last_1ms >= 1000) {
        last_1ms = current_time;

        // None for now
    }

    // Tasks to run every 10ms
    if (current_time - last_10ms >= 10000) {
        last_10ms = current_time;

        // Poll HAL
        MEMORY_BARRIER();
        MEMLNaut::Instance()->loop();
        MEMORY_BARRIER();

        // Refresh display
        if (display) {
            display->update();
        }

        // Blip
        static int blip_counter = 0;
        if (blip_counter++ > 100) {
            blip_counter = 0;
            float local_input_level = READ_VOLATILE(input_level);
            float local_input_pitch = READ_VOLATILE(input_pitch);
            //Serial.printf("Level: %f, Pitch: %f\n", local_input_level, local_input_pitch);
            Serial.println(".");
            // Blink LED
            digitalWrite(33, HIGH);
        } else {
            // Un-blink LED
            digitalWrite(33, LOW);
        }
    }
}

void setup1()
{
    while (!READ_VOLATILE(serial_ready)) {
        MEMORY_BARRIER();
        delay(1);
    }

    while (!READ_VOLATILE(interface_ready)) {
        MEMORY_BARRIER();
        delay(1);
    }

    // Create audio app with memory barrier protection
    {
        auto temp_audio_app = std::make_shared<CURRENT_AUDIO_APP>();
        temp_audio_app->Setup(AudioDriver::GetSampleRate(), interface);
        MEMORY_BARRIER();
        audio_app = temp_audio_app;
        MEMORY_BARRIER();
    }

    // Start audio driver
    AudioDriver::Setup(audio_app->GetDriverConfig());

    WRITE_VOLATILE(core_1_ready, true);
    while (!READ_VOLATILE(core_0_ready)) {
        MEMORY_BARRIER();
        delay(1);
    }

    Serial.println("Finished initialising core 1.");
}

void loop1()
{
    // Audio app parameter processing loop
    audio_app->loop();
}
