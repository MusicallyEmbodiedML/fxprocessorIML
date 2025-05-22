#include "src/memllib/PicoDefs.hpp"
#include "src/memllib/audio/AudioDriver.hpp"
#include "src/memllib/hardware/memlnaut/MEMLNaut.hpp"
#include <memory>
#include "src/memllib/interface/MIDIInOut.hpp"
#include "src/memllib/interface/UARTInput.hpp"

// Example apps and interfaces
#include "src/memllib/examples/IMLInterface.hpp"

#include "src/memllib/hardware/memlnaut/Display.hpp"


/**
 * @brief FX processor audio app
 *
 */

#include <cmath>
#include "src/daisysp/Effects/pitchshifter.h"
#include "src/memllib/synth/maximilian.h"
#include "src/memllib/audio/AudioAppBase.hpp"
#include "src/memllib/synth/OnePoleSmoother.hpp"

class FXProcessorAudioApp : public AudioAppBase
{
public:
    static constexpr size_t kN_Params = 7;

    FXProcessorAudioApp() : AudioAppBase(),
        setup_(false),
        smoother_(0.001, kSampleRate),
        target_params_(kN_Params, 0),
        smoothed_params_(kN_Params, 0),
        dl1_delay_time_(1.0f),
        dl1_feedback_(0.0f),
        dl1_wet_(0.0f),
        dl2_delay_time_(1.0f),
        dl2_feedback_(0.0f),
        dl2_wet_(0.0f) {}

    stereosample_t Process(const stereosample_t x) override
    {
        if (!setup_) {
            return { 0, 0 };
        }
        // Smooth parameters
        SmoothParams_();

        // Process audio
        float y = x.L + x.R;
        float yL, yR;
        float dry = y;

        // y = pitchshifter_.Process(y);
        y = y + delay_line_1_.play(y,
                                static_cast<size_t>(dl1_delay_time_),
                                dl1_feedback_) * dl1_wet_;
        // yR = y + delay_line_2_.play(y,
        //                        static_cast<size_t>(dl2_delay_time_),
        //                        dl2_feedback_) * dl2_wet_;

        // Apply dry/wet mix
        // y = 0.5f * yL + 0.5f * dry;
        // yR = 0.5f * yR + 0.5f * dry;
        stereosample_t ret { y, y };
        // PERIODIC_DEBUG(48000, {
        //     Serial.print("L: ");
        //     Serial.print(x.L);
        //     Serial.print(", R: ");
        //     Serial.println(x.R);
        // })
        return ret;
    }

    void Setup(float sample_rate, std::shared_ptr<InterfaceBase> interface) override
    {
        AudioAppBase::Setup(sample_rate, interface);
        // Additional setup code specific to FMSynthAudioApp
        // Set param smoothers
        smoother_.SetTimeMs(0.1f);
        // Pitch Shifter:
        pitchshifter_.Init(sample_rate);
        pitchshifter_.SetTransposition(0.f);

        // Setup finished
        setup_ = true;
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

    // DSP blocks:
    // - Pitch Shifter
    daisysp::PitchShifter pitchshifter_;
    // - Delay line 1
    static constexpr float kDelayLine1MaxTime = 0.5f;
    static constexpr size_t kDelayLine1Size = kDelayLine1MaxTime * kSampleRate;
    float dl1_delay_time_;
    float dl1_feedback_;
    float dl1_wet_;
    maxiDelayline<kDelayLine1Size> delay_line_1_;
    // - Delay line 2
    static constexpr float kDelayLine2MaxTime = 0.05f;
    static constexpr size_t kDelayLine2Size = kDelayLine2MaxTime * kSampleRate;
    float dl2_delay_time_;
    float dl2_feedback_;
    float dl2_wet_;
    maxiDelayline<kDelayLine2Size> delay_line_2_;

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

        // Assign smoothed parameters to their functions
        // Pitch Shifter:
        // - transposition
        float pitch_shift = LinearMap_(smoothed_params_[0], -12.f, 12.f);
        pitchshifter_.SetTransposition(pitch_shift);
        // Delay line 1:
        // - delay time
        dl1_delay_time_ = LinearMap_(smoothed_params_[1], 1.f,
                                     kDelayLine1MaxTime * kSampleRate - 1);
        // - feedback
        dl1_feedback_ = LinearMap_(smoothed_params_[2], 0.f, 0.95f);
        // - wet
        dl1_wet_ = SCurveMap_(smoothed_params_[3], 0.f, 1.f, 0.6f);
        // Delay line 2:
        // - delay time
        dl2_delay_time_ = LinearMap_(smoothed_params_[4], 1.f,
                                     kDelayLine2MaxTime * kSampleRate - 1);
        // - feedback
        dl2_feedback_ = LinearMap_(smoothed_params_[5], 0.f, 0.95f);
        // - wet
        dl2_wet_ = SCurveMap_(smoothed_params_[6], 0.f, 1.f, 0.6f);
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
const size_t kN_InputParams = 3;
const std::vector<size_t> kUARTListenInputs {};


void bind_interface(std::shared_ptr<CURRENT_INTERFACE> &interface)
{
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

    // Set up toggle switch callbacks
    MEMLNaut::Instance()->setTogA1Callback([interface] (bool state) {
        if (display) {
            display->post(state ? "Training mode" : "Inference mode");
        }
        interface->SetTrainingMode(state ? CURRENT_INTERFACE::TRAINING_MODE : CURRENT_INTERFACE::INFERENCE_MODE);
        if (display && state == false) {
            display->post("Model trained");
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
        MEMLNaut::Instance()->setJoyZCallback([interface] (float value) {
            interface->SetInput(2, value);
        });
    }
    // Set up other ADC callbacks
    MEMLNaut::Instance()->setRVZ1Callback([interface] (float value) {
        // Scale value from 0-1 range to 1-3000
        value = 1.0f + (value * 2999.0f);
        interface->SetIterations(static_cast<size_t>(value));
    });

    // Set up loop callback
    MEMLNaut::Instance()->setLoopCallback([interface] () {
        interface->ProcessInput();
    });

    MEMLNaut::Instance()->setRVGain1Callback([interface] (float value) {
        AudioDriver::setDACVolume(value);
        Serial.println(value);
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
    AudioDriver::Setup();

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
