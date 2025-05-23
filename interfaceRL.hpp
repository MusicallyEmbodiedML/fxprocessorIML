#ifndef INTERFACERL_HPP
#define INTERFACERL_HPP

#include "src/memllib/interface/InterfaceBase.hpp"

#include "src/memlp/MLP.h"
#include "src/memlp/ReplayMemory.hpp"
#include "src/memlp/OrnsteinUhlenbeckNoise.h"
#include <memory>
#include "sharedMem.hpp"

#include "src/memllib/PicoDefs.hpp"

#define RL_MEM __not_in_flash("rlmem")


struct trainRLItem {
    std::vector<float> state ;
    std::vector<float> action;
    float reward;
    std::vector<float> nextState;
};


class interfaceRL : public InterfaceBase
{
public:

    void setup(size_t n_inputs, size_t n_outputs)
    {
        const size_t nAudioAnalysisInputs = 0;
        const size_t nAllInputs = n_inputs + nAudioAnalysisInputs;
        InterfaceBase::setup(nAllInputs, n_outputs);


        stateSize = nAllInputs;
        actionSize = n_outputs;

        actor_layers_nodes = {
            stateSize + bias,
            10, 10,
            actionSize
        };

        critic_layers_nodes = {
            stateSize + actionSize + bias,
            10, 10,
            1
        };

        criticInput.resize(critic_layers_nodes[0]);
        actorControlInput.resize(actor_layers_nodes[0]);
        actorControlInput[actorControlInput.size()-1] = 1.f; // bias

        //init networks
        actor = std::make_shared<MLP<float> > (
            actor_layers_nodes,
            layers_activfuncs,
            loss::LOSS_MSE,
            use_constant_weight_init,
            constant_weight_init
        );

        actorTarget = std::make_shared<MLP<float> > (
            actor_layers_nodes,
            layers_activfuncs,
            loss::LOSS_MSE,
            use_constant_weight_init,
            constant_weight_init
        );

        critic = std::make_shared<MLP<float> > (
            critic_layers_nodes,
            layers_activfuncs,
            loss::LOSS_MSE,
            use_constant_weight_init,
            constant_weight_init
        );
        criticTarget = std::make_shared<MLP<float> > (
            critic_layers_nodes,
            layers_activfuncs,
            loss::LOSS_MSE,
            use_constant_weight_init,
            constant_weight_init
        );
    }

    void optimise() {
        constexpr size_t batchSize = 4;
        std::vector<trainRLItem> sample = replayMem.sample(batchSize);
        if (sample.size() == batchSize) {
            //run sample through critic target, build training set for critic net
            MLP<float>::training_pair_t ts;
            for(size_t i = 0; i < sample.size(); i++) {
                //---calculate y
                //--calc next-state-action pair
                //get next action from actorTarget given next state
                auto nextStateInput =  sample[i].nextState;
                nextStateInput.push_back(1.f); // bias
                actorTarget->GetOutput(nextStateInput, &actorOutput);

                //use criticTarget to estimate value of next action given next state
                for(size_t j=0; j < stateSize; j++) {
                    criticInput[j] = sample[i].nextState[j];
                }
                for(size_t j=0; j < actionSize; j++) {
                    criticInput[j+stateSize] = actorOutput[j];
                }
                criticInput[criticInput.size()-1] = 1.f; //bias

                criticTarget->GetOutput(criticInput, &criticOutput);

                //calculate expected reward
                const float y = sample[i].reward + (discountFactor *  criticOutput[0]);
                // std::cout << "[" << i << "]: y: " << y << std::endl;

                //use criticTarget to estimate value of next action given next state
                for(size_t j=0; j < stateSize; j++) {
                criticInput[j] = sample[i].state[j];
                }
                for(size_t j=0; j < actionSize; j++) {
                criticInput[j+stateSize] = sample[i].action[j];
                }
                criticInput[criticInput.size()-1] = 1.f; //bias

                ts.first.push_back(criticInput);
                ts.second.push_back({y});
            }

            //train the critic
            float loss = critic->Train(ts, learningRate, 1);

            //TODO: size limit to this log
            criticLossLog.push_back(loss);

            //update the actor

            //for each memory in replay memory sample, and get grads from critic
            std::vector<float> actorLoss(actionSize, 0.f);
            std::vector<float> gradientLoss= {1.f};

            for(size_t i = 0; i < sample.size(); i++) {
                //use criticTarget to estimate value of next action given next state
                for(size_t j=0; j < stateSize; j++) {
                criticInput[j] = sample[i].nextState[j];
                }
                for(size_t j=0; j < actionSize; j++) {
                criticInput[j+stateSize] = sample[i].action[j];
                }
                criticInput[criticInput.size()-1] = 1.f; //bias

                critic->CalcGradients(criticInput, gradientLoss);
                std::vector<float> l0Grads = critic->m_layers[0].GetGrads();

                for(size_t j=0; j < actionSize; j++) {
                 actorLoss[j] = l0Grads[j+stateSize];
                }
                delay(1);
            }

            float totalLoss = 0.f;
            for(size_t j=0; j < actorLoss.size(); j++) {
                actorLoss[j] /= sample.size();
                actorLoss[j] = -actorLoss[j];
                totalLoss += actorLoss[j];
            }
            // actorLossLog.push_back(actorLoss);
            // actorLoss = -actorLoss;
            // Serial.printf("Actor loss: %f\n", totalLoss);

            //back propagate the actor loss
            for(size_t i = 0; i < sample.size(); i++) {
                auto actorInput = sample[i].state;
                actorInput.push_back(bias);

                actor->ApplyLoss(actorInput, actorLoss, learningRate);
                delay(1);
            }

            // soft update the target networks
            criticTarget->SmoothUpdateWeights(critic, smoothingAlpha);
            actorTarget->SmoothUpdateWeights(actor, smoothingAlpha);
        }
    }

    void setState(const size_t index, float value) {
        actorControlInput[index] = value;
        newInput = true;
    }

    // void readAnalysisParameters() {
    //     //read analysis parameters
    //     actorControlInput[3] = READ_VOLATILE(sharedMem::f0);
    //     actorControlInput[4] = READ_VOLATILE(sharedMem::f1);
    //     actorControlInput[5] = READ_VOLATILE(sharedMem::f2);
    //     actorControlInput[6] = READ_VOLATILE(sharedMem::f3);
    //     PERIODIC_DEBUG(40, {
    //         Serial.println(actorControlInput[3]);
    //     })
    //     newInput = true;
    //     generateAction(true);
    // }

    void generateAction(bool donthesitate=false) {
        if (newInput || donthesitate) {
            newInput = false;
            std::vector<float> actorOutput;
            actorTarget->GetOutput(actorControlInput, &actorOutput);
            SendParamsToQueue(actorOutput);
            action = actorOutput;

            // for(size_t i=0; i < actorOutput.size(); i++) {
            //     const float noise = ou_noise.sample() * knobL;
            //     actorOutput[i] += noise;
            // }

        }
    }

    void optimiseSometimes() {
        if (optimiseCounter>=optimiseDivisor) {
            optimise();
            optimiseCounter=0;
        }else{
            optimiseCounter++;
        }
    }

    void storeExperience(float reward) {
        // readAnalysisParameters();
        std::vector<float> state = actorControlInput;
        float bpf0 = READ_VOLATILE(sharedMem::f0);

        //remove bias
        state.pop_back();

        for(size_t i=0; i < state.size(); i++) {
            Serial.printf("%f\t", state[i]);
        }
        Serial.println();
        trainRLItem trainItem = {state, action, reward, state};
        replayMem.add(trainItem, millis());
    }

    void randomiseTheActor()
    {
        actor->DrawWeights();
        actorTarget->DrawWeights();
    }

    void randomiseTheCritic()
    {
        critic->DrawWeights();
        criticTarget->DrawWeights();
    }

    void setOptimiseDivisor(size_t newDiv) {
        optimiseDivisor = newDiv;
    }

    void forgetMemory() {
        replayMem.clear();
    }


private:
    static constexpr size_t bias=1;

    size_t optimiseDivisor = 40;
    size_t optimiseCounter = 0;

    bool newInput=false;

    const std::vector<ACTIVATION_FUNCTIONS> layers_activfuncs = {
        RELU, RELU, SIGMOID
    };

    size_t stateSize;
    size_t actionSize;

    std::vector<size_t> actor_layers_nodes;
    std::vector<size_t> critic_layers_nodes;

    const bool use_constant_weight_init = false;
    const float constant_weight_init = 0;

    std::shared_ptr<MLP<float> > actor, actorTarget, critic, criticTarget;

    float discountFactor = 0.95;
    float learningRate = 0.005;
    float smoothingAlpha = 0.005;

    std::vector<float> action;

    ReplayMemory<trainRLItem> replayMem;

    std::vector<float> actorOutput, criticOutput;
    std::vector<float> criticInput;
    std::vector<float> actorControlInput;

    std::vector<float> criticLossLog, actorLossLog, log1;


};

#endif // INTERFACERL_HPP