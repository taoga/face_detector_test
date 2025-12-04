#ifndef ONNX_ITERATOR_H
#define ONNX_ITERATOR_H

#include <cstring>
#include <iostream>
#include "onnxruntime_cxx_api.h"
#include <vector>

struct tagLayer
{
    std::string         name;
    std::vector<int>    shape;
    size_t              shape_size;

};
using IOLayer = struct tagLayer;

class OnnxIterator
{
public:
    // Construction
    OnnxIterator(const std::string ModelPath, int device_num = 0, int threads = 1 )
    {
        init_engine_threads( threads, threads );
        init_onnx_model(ModelPath);
    }

    void reset_inputs()
    {
        for( auto &vec : vec_ort_inputs_ )
            vec.clear();
    }

    // !!! set inputs by order 0, 1 ... !!!
    template <typename T> int setInput( std::vector<T> &vf_in, const std::string &layer_name )
    {
        if( inputs_.size() == 0 ) return -1;

        // find layer
        size_t n_layer = 0;
        size_t i = 0;
        if( layer_name.length() > 0 ) {
            for( i = 0; i < inputs_.size(); i++ )
                if( inputs_[i].name == layer_name ) { n_layer = i; break; }
        }

        size_t  n_batchs = vf_in.size() / inputs_[n_layer].shape_size;

        //std::cout << "shape size:" << inputs_[n_layer].shape_size << " n_batchs:" << n_batchs << std::endl;

        if( n_batchs == 0 ) return -2;

        if( !n_layer )
        {
            for( auto &vec : vec_ort_inputs_ )
                vec.clear();

            vec_ort_inputs_.resize( n_batchs );
        }

        Ort::TypeInfo           info = session_->GetInputTypeInfo( n_layer );
        auto                    tensor_info = info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t>    i_shape = tensor_info.GetShape();
        std::vector<int>        usr_i_shape = input_shape( n_layer );


        if( i_shape.size() == usr_i_shape.size() )
            for( i = 0; i < usr_i_shape.size(); i++ )
                i_shape[i] = usr_i_shape[i];

        for( size_t n_batch = 0; n_batch < n_batchs; n_batch++ )
        {
            i_shape[0] = n_batch + 1; // batch size from 1 .. n_batchs
            Ort::Value input_ort = Ort::Value::CreateTensor<T>(
                memory_info_, const_cast<T*>(vf_in.data()), vf_in.size(), i_shape.data(), i_shape.size() );

            vec_ort_inputs_[n_batch].emplace_back(std::move(input_ort));
        }

        return 0;
    }

    template <typename T> T vectorProduct(const std::vector<T>& v){
        return accumulate(v.begin(),v.end(),1,std::multiplies<T>());
    }

    std::vector<Ort::Value>& predict( size_t n_batch  = 1 )
    {
        ort_outputs_.clear();
        if( n_batch > vec_ort_inputs_.size() ) return ort_outputs_;
        if( !session_ || inputs_.size() == 0 || outputs_.size() == 0 || vec_ort_inputs_[n_batch - 1].size() != inputs_.size() ) return ort_outputs_;
        // Infer
        ort_outputs_ = session_->Run(
            Ort::RunOptions{nullptr},
            *(sp_input_names_.get()), vec_ort_inputs_[n_batch - 1].data(), vec_ort_inputs_[n_batch - 1].size(),
            *(sp_output_names_.get()), outputs_.size());

        return ort_outputs_;
    }

    std::vector<int> output_shape(size_t idx = 0 ) { return ( idx >= outputs_.size() ) ? std::vector<int>{} : outputs_[idx].shape; }
    size_t output_size(size_t idx = 0 ) { return ( idx >= outputs_.size() ) ? 0 : outputs_[idx].shape_size; }

    std::vector<int> input_shape(size_t idx = 0 ) { return ( idx >= inputs_.size() ) ? std::vector<int>{} : inputs_[idx].shape; }
    template<typename T_VEC> int set_input_shape(T_VEC &modified_shape, size_t idx = 0 )
    {
        if ( idx < inputs_.size() && (inputs_[idx].shape.size() == modified_shape.size() || inputs_[idx].shape.size() == 0) )
        {
            inputs_[idx].shape.resize( modified_shape.size() );
            //std::cout << "input_shape #" << idx  << " name:" << inputs_[idx].name << std::endl;
            inputs_[idx].shape_size = 1;
            for( size_t n_el = 0 ; n_el < modified_shape.size(); n_el++ )
            {
                inputs_[idx].shape_size *= (modified_shape[n_el] == -1) ? 1 : modified_shape[n_el]; // elements in input shape
                inputs_[idx].shape[n_el] = modified_shape[n_el];

                //std::cout << modified_shape[n_el];
                //if( n_el < modified_shape.size() - 1 )
                //    std::cout << ", ";
                //else
                //    std::cout << std::endl;
            }
            //std::cout << "input elements: " << inputs_[idx].shape_size << std::endl;
            return 0;
        }
        return -1;
    }

protected:
    void init_engine_threads(int inter_threads, int intra_threads, bool b_use_cuda = true, int device_id = 0 )
    {
        // Optimization will take time and memory during startup
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options_.SetLogSeverityLevel(3);
        // CUDA options. If used.
        if( b_use_cuda )
        {
            session_options_.SetInterOpNumThreads(1);
            session_options_.SetIntraOpNumThreads(1);
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = device_id;  //GPU_ID
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive; // Algo to search for Cudnn
            cuda_options.arena_extend_strategy = 0;
            // May cause data race in some condition
            cuda_options.do_copy_in_default_stream = 0;
            session_options_.AppendExecutionProvider_CUDA(cuda_options); // Add CUDA options to session options
        }
        else
        {
            session_options_.SetIntraOpNumThreads(intra_threads);
            session_options_.SetInterOpNumThreads(inter_threads);
        }
    }

    void init_onnx_model(const std::string &model_path )
    {
        // Load model
        std::cout << "Load model:" << model_path << std::endl;
        session_ = std::make_shared<Ort::Session>(env_, model_path.c_str(), session_options_);
        // Get info
        std::cout << "inputs info:" << std::endl;
        size_t  i = 0, count = session_->GetInputCount();
        bool    b_enter_required = false;

        inputs_.resize(count);
        for(i = 0; i < count; i++ )
        {
            Ort::AllocatorWithDefaultOptions    ort_alloc;
            Ort::AllocatedStringPtr             tmp_name = session_->GetInputNameAllocated(i, ort_alloc);

            inputs_[i].shape.clear();
            inputs_[i].shape_size = 1;
            inputs_[i].name = tmp_name.get();

            std::cout << "#" << i << " :" << inputs_[i].name << std::endl;

            Ort::TypeInfo           info = session_->GetInputTypeInfo(i);
            auto                    tensor_info = info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t>    input_shape = tensor_info.GetShape();

            set_input_shape( input_shape, i);
        }
        if( b_enter_required ) std::cout << "required to enter shape!" << std::endl;

        std::cout << std::endl << "outputs info:" << std::endl;
        count = session_->GetOutputCount();
        outputs_.resize(count);
        for(i = 0; i < count; i++ )
        {
            Ort::AllocatorWithDefaultOptions ort_alloc;
            Ort::AllocatedStringPtr tmp_name = session_->GetOutputNameAllocated(i, ort_alloc);

            outputs_[i].name = tmp_name.get();
            outputs_[i].shape.clear();
            outputs_[i].shape_size = 1;

            std::cout << "#" << i << " :" << outputs_[i].name << std::endl;
            // init shape
            Ort::TypeInfo info = session_->GetOutputTypeInfo(i);
            auto tensor_info = info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> veci64_output_shape = tensor_info.GetShape();

            for( size_t n_el = 0; n_el < veci64_output_shape.size(); n_el++ )
            {
                outputs_[i].shape_size *= veci64_output_shape[n_el]; // elements in output shape
                outputs_[i].shape.push_back( veci64_output_shape[n_el] );
                std::cout << veci64_output_shape[n_el];
                if( n_el < veci64_output_shape.size() - 1 )
                    std::cout << ", ";
                else
                    std::cout << std::endl;
            }
            std::cout << "output elements: " << outputs_[i].shape_size << std::endl;
        }

        char**      p_input_names = new char*[inputs_.size()];
        char**      p_output_names = new char*[outputs_.size()];
        sp_input_names_ = std::make_shared<char**>(p_input_names);
        sp_output_names_ = std::make_shared<char**>(p_output_names);
        for( i = 0; i < inputs_.size(); i++ )
            p_input_names[i] = inputs_[i].name.data();
        for( i = 0; i < outputs_.size(); i++ )
            p_output_names[i] = outputs_[i].name.data();
    }

private:
    // OnnxRuntime resources
    Ort::Env                                env_;
    Ort::SessionOptions                     session_options_;
    std::shared_ptr<Ort::Session>           session_ = nullptr;
    Ort::AllocatorWithDefaultOptions        allocator_;
    Ort::MemoryInfo                         memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    //std::shared_ptr<Ort::MemoryInfo>        sp_memory_info_;
    // Onnx model
    // Inputs
    std::vector<IOLayer>                    inputs_;
    std::vector<std::vector<Ort::Value>>    vec_ort_inputs_;
    // Outputs
    std::vector<IOLayer>                    outputs_;
    std::vector<Ort::Value>                 ort_outputs_;

    std::shared_ptr<char**>                 sp_input_names_;
    std::shared_ptr<char**>                 sp_output_names_;
};

#endif // VAD_ITERATOR_H
