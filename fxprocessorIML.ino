#include "src/memllib/audio/AudioDriver.hpp"
#include "src/memllib/hardware/memlnaut/MEMLNaut.hpp"
#include <memory>
#include "src/memllib/interface/MIDIInOut.hpp"
#include "src/memllib/PicoDefs.hpp"
#include "src/memllib/interface/UARTInput.hpp"

// Example apps and interfaces
#include "src/memllib/examples/IMLInterface.hpp"

#include "src/memllib/hardware/memlnaut/display.hpp"


/**
 * @brief FX processor audio app
 *
 */

#include <cmath>
#include "src/daisysp/Effects/pitchshifter.h"
#include "src/memllib/synth/maximilian.h"
#include "src/memllib/audio/AudioAppBase.hpp"
#include "src/memllib/synth/OnePoleSmoother.hpp"

static constexpr float shift_amount = -3.f;

volatile float input_level = 0;

class FXProcessorAudioApp : public AudioAppBase
{
public:
    AudioDriver::codec_config_t GetDriverConfig() const override {
        return {
            .mic_input = false,
            .line_level = 3,
            .mic_gain_dB = 0,
            .output_volume = 0.95f
        };
    }

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
        WRITE_VOLATILE(input_level, std::abs(x.L) + std::abs(x.R));
        if (!setup_) {
            return { 0, 0 };
        }
        // Smooth parameters
        SmoothParams_();

        // Process audio
        float y = x.L, yL, yR;
        float dry = y;

        y = pitchshifter_.Process(y);
        yL = y + delay_line_1_.play(y,
                               static_cast<size_t>(dl1_delay_time_),
                               dl1_feedback_) * dl1_wet_;
        yR = y + delay_line_2_.play(y,
                               static_cast<size_t>(dl2_delay_time_),
                               dl2_feedback_) * dl2_wet_;

        // Apply dry/wet mix
        yL = 0.5f * yL + 0.5f * dry;
        yR = 0.5f * yR + 0.5f * dry;
        stereosample_t ret { yL, yR };
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
        pitchshifter_.SetTransposition(shift_amount);

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
        //pitchshifter_.SetTransposition(shift_amount);
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
std::shared_ptr<display> disp;

// Inter-core communication
volatile bool core_0_ready = false;
volatile bool core_1_ready = false;
volatile bool serial_ready = false;
volatile bool interface_ready = false;

// We're only bound to the joystick inputs (x, y, rotate)
const size_t kN_InputParams = 3;
const std::vector<size_t> kUARTListenInputs {};


void setup()
{
    Serial.begin(115200);
    //while (!Serial) {}
    Serial.println("Serial initialised.");
    WRITE_VOLATILE(serial_ready, true);

    // Setup board
    MEMLNaut::Initialize();
    pinMode(33, OUTPUT);
    disp = std::make_shared<display>();
    disp->setup();
    disp->post("MEML FX Unit");

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
        temp_interface->setup(total_input_params, CURRENT_AUDIO_APP::kN_Params, disp);
        MEMORY_BARRIER();
        temp_interface->SetMIDIInterface(midi_interf);
        MEMORY_BARRIER();
        interface = temp_interface;
        MEMORY_BARRIER();
    }
    WRITE_VOLATILE(interface_ready, true);

    // Bind interface after ensuring it's fully initialized
    interface->bindInterface();
    Serial.println("Bound interface to MEMLNaut.");
    interface->bindUARTInput(uart_input, kUARTListenInputs);
    Serial.println("Bound interface to UART input.");
    interface->bindMIDI(midi_interf);
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
        if (disp) {
            disp->update();
        }

        // Blip
        static int blip_counter = 0;
        if (blip_counter++ > 100) {
            blip_counter = 0;
            float local_input_level = READ_VOLATILE(input_level);
            Serial.println(local_input_level);
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
