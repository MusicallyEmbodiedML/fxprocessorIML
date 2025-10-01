#ifndef __TOM_AUDIO_APP_HPP__
#define __TOM_AUDIO_APP_HPP__

#include "src/memllib/audio/AudioDriver.hpp"
#include "src/memllib/PicoDefs.hpp"


extern volatile float input_level = 0;
extern volatile float input_pitch = 0;
extern volatile bool optimise_stop = false;
extern volatile bool dns_on = true;
extern volatile bool harmoniser_on = true;

extern maxiBiquad dns_hpf_;


class TomAudioApp : public AudioAppBase
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

    TomAudioApp() : AudioAppBase(),
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

    stereosample_t Process(const stereosample_t x) override
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

#endif  // __TOM_AUDIO_APP_HPP__
