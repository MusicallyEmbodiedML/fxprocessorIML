bool core1_disable_systick = true;
bool core1_separate_stack = true;

#include "src/memllib/audio/AudioDriver.hpp"
#include "src/memllib/hardware/memlnaut/MEMLNaut.hpp"
#include <memory>
#include "src/memllib/interface/MIDIInOut.hpp"
#include "src/memllib/PicoDefs.hpp"
#include "src/memllib/interface/UARTInput.hpp"

// Example apps and interfaces
#include "src/memllib/examples/InterfaceRL.hpp"

#include "src/memllib/hardware/memlnaut/Pins.hpp"

#include <new> // for placement new
#include "hardware/structs/bus_ctrl.h"

#define APP_SRAM __not_in_flash("app")

/**
 * @brief FX processor audio app
 *
 */

#include <cmath>
#include <cstring>
#include "src/memllib/synth/maximilian.h"
#include "src/memllib/audio/AudioAppBase.hpp"
#include "src/memllib/synth/OnePoleSmoother.hpp"

volatile float input_level = 0;
volatile float input_pitch = 0;
//volatile bool randomise_actor = false;
volatile bool optimise_stop = false;
volatile bool dns_on = true;
volatile bool harmoniser_on = true;

AUDIO_MEM maxiBiquad dns_hpf_;

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

    static constexpr size_t kPatternLength = 8;

    struct Params {
        float f_drift;
        float env_release;
        float dns_ratio;
        float cutoffs[kPatternLength];
        float resos[kPatternLength];
        float oscXFade0;
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
        smoothed_params_(kN_Params, 0),
        pattern_idx_(0) {}

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
        for (size_t n = 0; n < kFreqScalingsSize; n++) {
            osc_[n].UpdateParams();
        }

        // Downsampler
        // dns_lpf_.set(maxiBiquad::LOWPASS,
        //              2000.f,
        //              5.f, 0);
        dns_hpf_.set(maxiBiquad::HIGHPASS,
                     40.f,
                     0.707f, 0);

        // Default parameters
        Params default_params {
            .f_drift = 0.97,
            .env_release = 100,
            .dns_ratio = 15,
            .cutoffs = { 100, 200, 300, 400, 500, 600, 700, 800 },
            .resos = { 1, 2, 3, 4, 5, 6, 7, 8 }
        };
        params_ = default_params;
        pattern_idx_ = 0;

        // Setup finished
        setup_ = true;
    }

    __force_inline stereosample_t ProcessInline(const stereosample_t x)
    {
        // WRITE_VOLATILE(input_level, std::abs(x.L) + std::abs(x.R));
        if (!setup_) {
            return { 0.f, 0.f };
        }
        // Smooth parameters
        SmoothParams_();

        float dry = x.L;
        env.play(dry);
        // Pitch detection
        float detect = bandpass_low.play(dry);
        detect = bandpass_high.play(detect);
        float pitch = pitch_detector.process(detect);
        // WRITE_VOLATILE(input_pitch, pitch);
        // if (pitch < freq_range.min || pitch > freq_range.max) {
        //     pitch = 0;
        // }

        // Synth
        float synth = 0;
        float scalings[kFreqScalingsSize] = {
            params_.f_drift,
            1.f/params_.f_drift
        };
        float oscamps[kFreqScalingsSize] = {
            sqrtf(params_.oscXFade0), sqrtf(1.f-params_.oscXFade0)
        };
        for (size_t n = 0; n < kFreqScalingsSize; n++) {
            float oscFreq = pitch * scalings[n];
            float oscSine = osc_[n].sinewave(oscFreq) * oscamps[n];
            // float oscSine = osc_[n].sinewave(oscFreq) * params_.oscXFade0;
            // float oscSaw = oscSaw_[n].square(oscFreq) * (1.f - params_.oscXFade0);
            // float oscMix = oscSine + oscSaw;
            synth += oscSine * kFreqScalingsVol;
        }

        // Decimation
        float decim = dns_.play(dry, kSampleRate / params_.dns_ratio);
        static float __not_in_flash("dsp") decim_prev = 0.f;
        if (decim != decim_prev) {
            decim_prev = decim;
            dns_lpf_.setParams(params_.cutoffs[pattern_idx_],
                params_.resos[pattern_idx_]);
            if (++pattern_idx_ >= kPatternLength) {
                pattern_idx_ = 0;
            }
        }
        decim = dns_lpf_.play(decim, 1.0f, 0, 0, 0);
        decim = dns_hpf_.play(decim);
        decim *= 10.f;
        decim = tanhf(decim);
        //decim = quant_.play(decim, 16);

        // Output
        float out_env = env.getEnv();
        out_env *= 6.f;
        out_env = powf(out_env, 1.8f);
        out_env = tanhf(out_env);
        // if (out_env < 0.01) {
        //     out_env = 0;
        // }
        float decim_gain = (dns_on) ? 0.5f : 0.f;
        float harmoniser_gain = (harmoniser_on) ? 1.f : 0.f;
        out_env *= harmoniser_gain;
        float yL = synth * out_env + decim * decim_gain;
        yL = tanhf(yL);
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
    static constexpr size_t kFreqScalingsSize = 2;
    static constexpr float kFreqScalingsVol =
        1.f/static_cast<float>(kFreqScalingsSize);
    maxiOsc osc_[kFreqScalingsSize];
    maxiOsc oscSaw_[kFreqScalingsSize];

    // Downsampler
    // - downsample
    maxiDownSample dns_;
    // - decimate
    maxiBitQuant quant_;
    maxiSVF dns_lpf_;
    // Loop
    size_t pattern_idx_;

    float oscXFadeGain0=1.0f, osc1XFadeGain1=1.0f;

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
     * @brief Double linear mapping function with intermediate point
     *
     * @param x float between 0 and 1
     * @param out_min output value at x=0
     * @param mid_x x coordinate of intermediate point (between 0 and 1)
     * @param mid_y output value at x=mid_x
     * @param out_max output value at x=1
     * @return float Interpolated value
     */
    static __attribute__((always_inline)) float DoubleLinearMapping_(float x, float out_min, float mid_x, float mid_y, float out_max)
    {
        // Branchless clamp of x to [0,1]
        x = x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);

        // Pre-compute slopes and determine which segment to use
        const float slope1 = (mid_y - out_min) / mid_x;
        const float slope2 = (out_max - mid_y) / (1.0f - mid_x);

        // Branchless segment selection using step function
        const float t = x <= mid_x ? 0.0f : 1.0f;

        // First segment: out_min + x * slope1
        // Second segment: mid_y + (x - mid_x) * slope2
        // Use fma for better precision and potential hardware acceleration
        return t * (mid_y + fma(x - mid_x, slope2, 0.0f)) +
            (1.0f - t) * fma(x, slope1, out_min);
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

    /**
 * @brief Exponential mapping function optimized for frequency scaling (pow(2,x))
 *
 * @param x float between 0 and 1
 * @param out_min minimum output value (frequency)
 * @param out_max maximum output value (frequency)
 * @return float Mapped frequency value
 */
static inline __attribute__((always_inline)) float ExpMap_(float x, float out_min, float out_max) {
    // Clamp x using branchless min/max
    x = x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);

    // Calculate log2 range for frequency scaling
    const float log2_min = std::log2(out_min);
    const float log2_max = std::log2(out_max);

    // Linear interpolation in log space
    const float log2_val = log2_min + (x * (log2_max - log2_min));

    // Fast pow2 approximation using IEEE float bit manipulation
    union {
        float f;
        int32_t i;
    } u;

    const float c_log2 = 1.442695040f; // 1/ln(2)
    const int32_t offset = 0x3f800000; // IEEE float 1.0 in hex

    // Calculate 2^x using bit manipulation
    u.i = static_cast<int32_t>(log2_val * (1 << 23)) + offset;

    return u.f;
}

    void SmoothParams_() {
        smoother_.Process(target_params_.data(), smoothed_params_.data());

        auto param_ptr = smoothed_params_.data();
        // Assign smoothed parameters to their functions
        // params_.f_drift = DoubleLinearMapping_(*param_ptr++, 0.60, 0.5, 0.95, 0.995);
        params_.f_drift = DoubleLinearMapping_(*param_ptr++, 0.40, 0.9, 0.95, 0.995);
        params_.dns_ratio = SCurveMap_(*param_ptr++, 4, 12, 0.3);
        params_.env_release = LinearMap_(*param_ptr++, 10, 500);
        env.setRelease(params_.env_release);
        // Map remaining params to the pattern
        std::memcpy(params_.cutoffs, target_params_.data()+3,
                    sizeof(float)*kPatternLength);
        std::memcpy(params_.resos, target_params_.data()+3+kPatternLength,
                    sizeof(float)*kPatternLength);
        for (unsigned int n = 0; n < kPatternLength; n++) {
            // Scale cutoff
            params_.cutoffs[n] = ExpMap_(params_.cutoffs[n], 500, 5000);
            // Scale reso
            params_.resos[n] = LinearMap_(params_.resos[n], 0.707, 8);
        }
        params_.oscXFade0 = LinearMap_(*param_ptr++, 0, 1);
    }
};


/******************************* */

class TomRLInterface : public InterfaceRL
{
public:
    void bind_RL_interface(bool disable_joystick = false) override
    {
    #if 1
        MEMLNaut::Instance()->setTogB1Callback([this] (bool value) {
            if (!value) return;
            this->_forget_replay_mem_interf();
        });
    #endif

        // Set up ADC callbacks
        MEMLNaut::Instance()->setJoyXCallback([this] (float value) {
            this->setState(0, value);
        });
        MEMLNaut::Instance()->setJoyYCallback([this] (float value) {
            this->setState(1, value);
        });
        // MEMLNaut::Instance()->setJoyZCallback([interface] (float value) {
        //     interface->setState(2, value);
        // });

        MEMLNaut::Instance()->setRVX1Callback([this] (float value) {
            this->setOptimiseDivisorInterf(value);
        });
        MEMLNaut::Instance()->setRVX1Callback([this](float value) { // scr_ref no longer captured directly
            this->setOptimiseDivisorInterf(value);
        });

        MEMLNaut::Instance()->setRVY1Callback([this](float value) {
            // this->setRewardScaleInterf(value);
            this->setLRScale(value);
        });

        MEMLNaut::Instance()->setRVZ1Callback([this](float value) { // scr_ref no longer captured directly
            setNoiseLevel(value);
        });

        // Set up loop callback
        MEMLNaut::Instance()->setLoopCallback([this] () {
            bool optimise_stop_local = READ_VOLATILE(optimise_stop);
            if (!optimise_stop_local) {
                this->optimiseSometimes();
            }
            this->generateAction();
        });
    #if 1
        MEMLNaut::Instance()->setTogA1Callback([] (bool value) {
            if (!value) return;
            Serial.printf("%s RL optimisation.\n", (optimise_stop) ? "Starting" : "Stopping");
            bool optimise_stop_local = READ_VOLATILE(optimise_stop);
            optimise_stop_local = !optimise_stop_local;
            WRITE_VOLATILE(optimise_stop, optimise_stop_local);
            if (optimise_stop_local) {
                //TODO
                //interface->saveNetworks();
            }
        });
    #endif
        MEMLNaut::Instance()->setMomA2Callback([this] () {
            this->trigger_like();
        });
        MEMLNaut::Instance()->setTogA2Callback([this] (bool value) {
            if (!value) return;
            this->trigger_dislike();
        });

    #if 1
        MEMLNaut::Instance()->setTogB2Callback([this] (bool value) {
            if (!value) return;
            this->trigger_randomiseRL();
        });
    #endif
        MEMLNaut::Instance()->setJoySWCallback([this] (bool value) {
            if (!value) return;
            bool dns_on_local = READ_VOLATILE(dns_on);
            bool harmoniser_on_local = READ_VOLATILE(harmoniser_on);
            if (harmoniser_on_local && dns_on_local) {
                dns_on_local = false;
                harmoniser_on_local = true;
            } else if (harmoniser_on_local && !dns_on_local) {
                dns_on_local = true;
                harmoniser_on_local = false;
            } else {
                harmoniser_on_local = true;
                dns_on_local = true;
            }
            WRITE_VOLATILE(dns_on, dns_on_local);
            WRITE_VOLATILE(harmoniser_on, harmoniser_on_local);
            Serial.printf("DNS %s Harm %s.\n", dns_on_local ? "off" : "on", harmoniser_on_local ? "off" : "on");
        });

        setOptimiseDivisor(2);
    }
};



// Global objects
using CURRENT_AUDIO_APP = FXProcessorAudioApp;
//using CURRENT_INTERFACE = IMLInterface;
using CURRENT_INTERFACE = TomRLInterface;

#define APP_SRAM __not_in_flash("app")

static constexpr char APP_NAME[] = "-- FXProcessor Tom --";

// Statically allocated, properly aligned storage in AUDIO_MEM for objects
alignas(FXProcessorAudioApp) char AUDIO_MEM audio_app_mem[sizeof(FXProcessorAudioApp)];

uint32_t get_rosc_entropy_seed(int bits) {
    uint32_t seed = 0;
    for (int i = 0; i < bits; ++i) {
        // Wait for a bit of time to allow jitter to accumulate
        busy_wait_us_32(5);
        // Pull LSB from ROSC rand output
        seed <<= 1;
        seed |= (rosc_hw->randombit & 1);
    }
    return seed;
}

std::shared_ptr<CURRENT_INTERFACE> APP_SRAM interface;
std::shared_ptr<CURRENT_AUDIO_APP> APP_SRAM audio_app;
// std::shared_ptr<MIDIInOut> midi_interf;
// std::shared_ptr<UARTInput> uart_input;
// std::shared_ptr<Display> display;

// Inter-core communication
volatile bool core_0_ready = false;
volatile bool core_1_ready = false;
volatile bool serial_ready = false;
volatile bool interface_ready = false;

// We're only bound to the joystick inputs (x, y)
const size_t kN_InputParams = 2;


void setup()
{
    // FILE *fp = fopen("/thisfilelivesonflash.txt", "w");
    // fprintf(fp, "Hello!\n");
    // fclose(fp);

    bus_ctrl_hw->priority = BUSCTRL_BUS_PRIORITY_DMA_W_BITS |
        BUSCTRL_BUS_PRIORITY_DMA_R_BITS | BUSCTRL_BUS_PRIORITY_PROC1_BITS;

    uint32_t seed = get_rosc_entropy_seed(32);
    srand(seed);

    Serial.begin(115200);
    //while (!Serial) {}
    Serial.println("Serial initialised.");
    WRITE_VOLATILE(serial_ready, true);

    // Setup board
    MEMLNaut::Initialize();
    pinMode(33, OUTPUT);
    digitalWrite(33,0);
    {
        auto temp_interface = std::make_shared<CURRENT_INTERFACE>();
        temp_interface->setup(kN_InputParams, CURRENT_AUDIO_APP::kN_Params);
        MEMORY_BARRIER();
        interface = temp_interface;
        MEMORY_BARRIER();
    }
    // Setup interface with memory barrier protection
    WRITE_VOLATILE(interface_ready, true);
    // Bind interface after ensuring it's fully initialized
    interface->bind_RL_interface(true);
    Serial.println("Bound interface to MEMLNaut.");

    //scr_ptr->post(APP_NAME);
    //add_repeating_timer_ms(-39, displayUpdate, NULL, &timerDisplay);
    std::shared_ptr<MessageView> helpView = std::make_shared<MessageView>("Help");
    helpView->post(APP_NAME);
    helpView->post("TA: Down: Forget replay memory");
    helpView->post("MA: Up: Randomise actor");
    helpView->post("MA: Down: Randomise critic");
    helpView->post("MB: Up: Positive reward");
    helpView->post("MB: Down: Negative reward");
    helpView->post("Y: Optimisation rate");
    helpView->post("Z: OU noise");
    helpView->post("Joystick: Explore");
    MEMLNaut::Instance()->disp->AddView(helpView);

    MEMLNaut::Instance()->addSystemInfoView();

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
    // {
    //     // Poll the UART input
    //     uart_input->Poll();
    //     // Poll the MIDI interface
    //     midi_interf->Poll();
    // }

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
        //pollButtons();


        // bool randomise_local = READ_VOLATILE(randomise_actor);
        // if (randomise_local) {
        //     //interface->randomiseTheActor();
        //     Serial.println("Actor randomised");
        // }

        // Refresh display
        // if (display) {
        //     display->update();
        // }

        // Blip
        static int blip_counter = 0;
        if (blip_counter++ > 100) {
            blip_counter = 0;
            float local_input_level = READ_VOLATILE(input_level);
            float local_input_pitch = READ_VOLATILE(input_pitch);
            //Serial.printf("Level: %f, Pitch: %f\n", local_input_level, local_input_pitch);
            //Serial.print("Free heap: ");
            //Serial.print(rp2040.getFreeHeap());
            //Serial.println(" bytes.");
            bool dsp_overload_local = READ_VOLATILE(dsp_overload);
            if (dsp_overload_local) {
                Serial.println("OVERLOAD");
            } else {
                Serial.println(".");
            }
            // Blink LED
            digitalWrite(33, HIGH);
        } else {
            // Un-blink LED
            digitalWrite(33, LOW);
        }
    }
}


void AUDIO_FUNC(audio_block_callback)(float in[][kBufferSize], float out[][kBufferSize], size_t n_channels, size_t n_frames)
{
    digitalWrite(Pins::LED_TIMING, HIGH);
    for (size_t i = 0; i < n_frames; ++i) {

        stereosample_t x {
            in[0][i],
            in[1][i]
        }, y;

        // Audio processing
        if (audio_app) {
            y = audio_app->ProcessInline(x);
        } else {
            y = x; // Pass through if audio_app is not ready
            y.L *= y.L;
            y.R *= y.R;
        }

        out[0][i] = y.L;
        out[1][i] = y.R;
    }
    digitalWrite(Pins::LED_TIMING, LOW);

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

    // Create audio app using placement-new into static buffer and custom deleter
    {
        CURRENT_AUDIO_APP* audio_raw = new (audio_app_mem) CURRENT_AUDIO_APP();
        std::shared_ptr<InterfaceBase> selectedInterface = std::dynamic_pointer_cast<InterfaceBase>(interface);

        audio_raw->Setup(AudioDriver::GetSampleRate(), selectedInterface);

        // shared_ptr with custom deleter calling only the destructor (control block still allocates)
        auto audio_deleter = [](CURRENT_AUDIO_APP* p) { if (p) p->~CURRENT_AUDIO_APP(); };
        std::shared_ptr<CURRENT_AUDIO_APP> temp_audio_app(audio_raw, audio_deleter);

        MEMORY_BARRIER();
        audio_app = temp_audio_app;
        MEMORY_BARRIER();
    }

    // Start audio driver
    AudioDriver::SetBlockCallback(audio_block_callback);
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
