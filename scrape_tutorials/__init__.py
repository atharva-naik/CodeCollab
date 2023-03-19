#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# code for gathering, processing and indexing tutorials
# encode task decompositions

import json

SOURCE_TO_BASE_URLS = {
    "seaborn": "https://seaborn.pydata.org/tutorial",
    "pandas": {
        "Modern Pandas": "https://tomaugspurger.github.io/posts/modern-1-intro/",
        "Method Chaining": "https://tomaugspurger.github.io/posts/method-chaining/",
        "Indexes": "https://tomaugspurger.github.io/posts/modern-3-indexes/",
        "Fast Pandas": "https://tomaugspurger.github.io/posts/modern-4-performance/",
        "Tidy Data": "https://tomaugspurger.github.io/posts/modern-5-tidy/",
        "Visualization": "https://tomaugspurger.github.io/posts/modern-6-visualization/",
        "Time Series": "https://tomaugspurger.github.io/posts/modern-7-timeseries/",
        "Scaling": "https://tomaugspurger.github.io/posts/modern-8-scaling/", 
    },
    "numpy": {
        "NumPy Features": {
            "Linear algebra on n-dimensional arrays": "https://numpy.org/numpy-tutorials/content/tutorial-svd.html",
            "Saving and sharing your NumPy arrays": "https://numpy.org/numpy-tutorials/content/save-load-arrays.html",
            "Masked Arrays": "https://numpy.org/numpy-tutorials/content/tutorial-ma.html",
        },
        "NumPy Applications": {
            "Determining Moore’s Law with real data in NumPy": "https://numpy.org/numpy-tutorials/content/mooreslaw-tutorial.html",
            "Deep learning on MNIST": "https://numpy.org/numpy-tutorials/content/tutorial-deep-learning-on-mnist.html",
            "X-ray image processing": "https://numpy.org/numpy-tutorials/content/tutorial-x-ray-image-processing.html",
            "Determining Static Equilibrium in NumPy": "https://numpy.org/numpy-tutorials/content/tutorial-static_equilibrium.html",
            "Plotting Fractals": "https://numpy.org/numpy-tutorials/content/tutorial-plotting-fractals.html",
            "Analyzing the impact of the lockdown on air quality in Delhi, India": "https://numpy.org/numpy-tutorials/content/tutorial-air-quality-analysis.html",
        },
        "Articles": {
            "Deep reinforcement learning with Pong from pixels": "https://numpy.org/numpy-tutorials/content/tutorial-deep-reinforcement-learning-with-pong-from-pixels.html",
            "Sentiment Analysis on notable speeches of the last decade": "https://numpy.org/numpy-tutorials/content/tutorial-nlp-from-scratch.html",
        }
    },
    "torch": {
        "PyTorch Recipes": {
            "Loading Data in PyTorch": "https://pytorch.org/tutorials/recipes/recipes/loading_data_recipe.html",
            "Defining a Neural Network in PyTorch": "https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html",
            "What is a state_dict in PyTorch": "https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html",
            "Developing Custom PyTorch Dataloaders": "https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html",
            "Model Interpretability using Captum": "https://pytorch.org/tutorials/recipes/recipes/Captum_Recipe.html",
            "Dynamic Quantization": "https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html",
            "Saving and loading models across devices in PyTorch": "https://pytorch.org/tutorials/recipes/recipes/save_load_across_devices.html",
            "Saving and loading a general checkpoint in PyTorch": "https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html",
            "Saving and loading models for inference in PyTorch": "https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html",
            "Saving and loading multiple models in one file using PyTorch": "https://pytorch.org/tutorials/recipes/recipes/saving_multiple_models_in_one_file.html",
            "Warmstarting models using parameters from different model": "https://pytorch.org/tutorials/recipes/recipes/warmstarting_model_using_parameters_from_a_different_model.html",
            "Zeroing out gradients": "https://pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html",
            "PyTorch Mobile Performance Recipes": "https://pytorch.org/tutorials/recipes/mobile_perf.html",
            "Automatic Mixed Precision": "https://pytorch.org/tutorials/recipes/amp_recipe.html",
            "Changing default device": "https://pytorch.org/tutorials/recipes/recipes/changing_default_device.html",   
            "How to use TensorBoard with PyTorch": "https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html",
            "Performance Tuning Guide": "https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html",
            "Timer quick start": "https://pytorch.org/tutorials/recipes/recipes/timer_quick_start.html",
            "PyTorch Profiler": "https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html",
            "PyTorch Benchmark": "https://pytorch.org/tutorials/recipes/recipes/benchmark.html#sphx-glr-recipes-recipes-benchmark-py",
        },
        "Introduction to PyTorch": {
            "Learn the Basics": "https://pytorch.org/tutorials/beginner/basics/intro.html",
            "Quickstart": "https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html",
            "Tensors": "https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html",
            "Datasets & DataLoaders": "https://pytorch.org/tutorials/beginner/basics/data_tutorial.html",
            "Transforms": "https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html",
            "Build the Neural Network": "https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html",
            "Automatic Differentiation with torch.autograd": "https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html",
            "Optimizing Model Parameters": "https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html",
            "Save and Load the Model": "https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html", 
        },
        "Learning PyTorch": {
            "Deep Learning with PyTorch: A 60 Minute Blitz": "https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html",
            "Learning PyTorch with Examples": "https://pytorch.org/tutorials/beginner/pytorch_with_examples.html",
            "What is torch.nn really?": "https://pytorch.org/tutorials/beginner/nn_tutorial.html",
            "Visualizing Models, Data, and Training with TensorBoard": "https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html",
        },
        "Image and Video": {
            "TorchVision Object Detection Finetuning Tutorial": "https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html",
            "Transfer Learning for Computer Vision Tutorial": "https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html",
            "Adversarial Example Generation": "https://pytorch.org/tutorials/beginner/fgsm_tutorial.html",
            "DCGAN Tutorial": "https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html",
            "Spatial Transformer Networks Tutorial": "https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html",
            "Optimizing Vision Transformer Model for Deployment": "https://pytorch.org/tutorials/beginner/vt_tutorial.html",
        },
        "Audio": {
            "Audio I/O": "https://pytorch.org/tutorials/beginner/audio_io_tutorial.html",
            "Audio Resampling": "https://pytorch.org/tutorials/beginner/audio_resampling_tutorial.html",
            "Audio Data Augmentation": "https://pytorch.org/tutorials/beginner/audio_data_augmentation_tutorial.html",
            "Audio Feature Extractions": "https://pytorch.org/tutorials/beginner/audio_feature_extractions_tutorial.html",
            "Audio Feature Augmentation": "https://pytorch.org/tutorials/beginner/audio_feature_augmentation_tutorial.html",
            "Audio Datasets": "https://pytorch.org/tutorials/beginner/audio_datasets_tutorial.html",
            "Speech Recognition with Wav2Vec2": "https://pytorch.org/tutorials/intermediate/speech_recognition_pipeline_tutorial.html",
            "Text-to-speech with Tacotron2": "https://pytorch.org/tutorials/intermediate/text_to_speech_with_torchaudio.html",
            "Forced Alignment with Wav2Vec2": "https://pytorch.org/tutorials/intermediate/forced_alignment_with_torchaudio_tutorial.html", 
        },
        "Text": {
            "Language Modeling with nn.Transformer and TorchText": "https://pytorch.org/tutorials/beginner/transformer_tutorial.html",
            "Fast Transformer Inference with Better Transformer": "https://pytorch.org/tutorials/beginner/bettertransformer_tutorial.html",
            "NLP From Scratch: Classifying Names with a Character-Level RNN": "https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html",
            "NLP From Scratch: Generating Names with a Character-Level RNN": "https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html",
            "Text classification with the torchtext library": "https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html",
            "Language Translation with nn.Transformer and torchtext": "https://pytorch.org/tutorials/beginner/translation_transformer.html",
        },
        "Reinforcement Learning": {
           "Reinforcement Learning (DQN) Tutorial": "https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html",
           "Reinforcement Learning (PPO) with TorchRL Tutorial": "https://pytorch.org/tutorials/intermediate/reinforcement_ppo.html", 
           "Train a Mario-playing RL Agent": "https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html",
        },
        "Deploying PyTorch Models in Production": {
            "Deploying PyTorch in Python via a REST API with Flask": "https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html",
            "Introduction to TorchScript": "https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html",
            "Loading a TorchScript Model in C++": "https://pytorch.org/tutorials/advanced/cpp_export.html",
            "(optional) Exporting a Model from PyTorch to ONNX and Running it using ONNX Runtime": "https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html",
            "Real Time Inference on Raspberry Pi 4 (30 fps!)": "https://pytorch.org/tutorials/intermediate/realtime_rpi.html",
        },
        "Code Transforms with FX": {
            "(beta) Building a Convolution/Batch Norm fuser in FX": "https://pytorch.org/tutorials/intermediate/fx_conv_bn_fuser.html",
            "(beta) Building a Simple CPU Performance Profiler with FX": "https://pytorch.org/tutorials/intermediate/fx_profiling_tutorial.html",
        },
        "Frontend APIs": {
            "(beta) Channels Last Memory Format in PyTorch": "https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html",
            "Forward-mode Automatic Differentiation (Beta)": "https://pytorch.org/tutorials/intermediate/forward_ad_usage.html",
            "Jacobians, Hessians, hvp, vhp, and more: composing function transforms": "https://pytorch.org/tutorials/intermediate/jacobians_hessians.html",
            "Model ensembling": "https://pytorch.org/tutorials/intermediate/ensembling.html",
            "Per-sample-gradients": "https://pytorch.org/tutorials/intermediate/per_sample_grads.html",
            "Using the PyTorch C++ Frontend": "https://pytorch.org/tutorials/advanced/cpp_frontend.html",
            "Dynamic Parallelism in TorchScript": "https://pytorch.org/tutorials/advanced/torch-script-parallelism.html",
            "Autograd in C++ Frontend": "https://pytorch.org/tutorials/advanced/cpp_autograd.html",
        },
        "Extending PyTorch": {
            "Double Backward with Custom Functions": "https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html",
        }
    },
}