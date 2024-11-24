use wgpu::util::DeviceExt;
use bytemuck;

fn compute_cosine_similarity(target: &[f32], vectors: &[Vec<f32>]) -> Vec<f32> {
    // The main function that sets up wgpu and runs the compute shader

    // First, create an async block and block on it (since wgpu initialization is async)
    let similarities = pollster::block_on(async {
        // Step 1: Initialize wgpu (adapter, device, queue)
        let instance = wgpu::Instance::default();

        // Since we don't have a window, we set compatible_surface to None
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None, // No surface since we're doing compute
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find an appropriate adapter");

        // Request a device and queue from the adapter
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(), // We don't need any special features
                    limits: wgpu::Limits::default(),
                    label: None,
                },
                None, // Trace path
            )
            .await
            .expect("Failed to create device");

        // Step 2: Prepare buffers and data

        // Flatten the vectors into a single buffer
        let vector_length = target.len() as u32;
        let num_vectors = vectors.len() as u32;

        // Check that all vectors have the same length
        for v in vectors {
            assert_eq!(v.len(), vector_length as usize, "All vectors must have the same length");
        }

        // Create buffers

        // Target vector buffer
        let target_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Target Buffer"),
            contents: bytemuck::cast_slice(&target),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Vectors buffer
        let vectors_data: Vec<f32> = vectors.iter().flat_map(|v| v.clone()).collect();
        let vectors_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vectors Buffer"),
            contents: bytemuck::cast_slice(&vectors_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Output buffer to store the results
        let output_buffer_size = num_vectors as usize * std::mem::size_of::<f32>();
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: output_buffer_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Parameters buffer (uniform buffer)
        // We need to pass the vector_length and num_vectors to the shader
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            vector_length: u32,
            num_vectors: u32,
        }
        let params = Params {
            vector_length,
            num_vectors,
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Step 3: Create compute shader

        // Load the WGSL shader code
        let shader_source = include_str!("cosine_similarity.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cosine Similarity Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Step 4: Set up pipeline

        // Create the bind group layout to describe the resources the shader can access
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[
                // Binding 0: Target vector
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: Vectors buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 2: Output buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 3: Params buffer (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create the bind group to bind the actual buffers
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                // Binding 0: Target vector
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: target_buffer.as_entire_binding(),
                },
                // Binding 1: Vectors buffer
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: vectors_buffer.as_entire_binding(),
                },
                // Binding 2: Output buffer
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                // Binding 3: Params buffer
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create the compute pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "main",
        });

        // Step 5: Dispatch compute shader

        // Create a command encoder to record commands
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });

        {
            // Begin a compute pass
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
            });
            // Set the pipeline and bind group
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            // Dispatch workgroups (one per vector)
            compute_pass.dispatch_workgroups(num_vectors, 1, 1);
        }

        // Step 6: Copy output buffer to CPU accessible buffer

        // Create a buffer on the CPU to receive the data
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_buffer_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy the results from the GPU output buffer to the staging buffer
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            output_buffer_size as u64,
        );

        // Submit the commands to the GPU
        queue.submit(Some(encoder.finish()));

        // Wait for the GPU to finish executing
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        device.poll(wgpu::Maintain::Wait);
        receiver.receive().await.unwrap().unwrap();

        // Read the data from the buffer
        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

        // Cleanup
        drop(data);
        staging_buffer.unmap();

        result
    });

    similarities
}

fn main() {
    // Prepare the data
    let target = vec![1.0f32, 2.0, 3.0, 4.0]; // Example target vector
    let vectors = vec![
        vec![4.0f32, 3.0, 2.0, 1.0],
        vec![1.0f32, 0.0, 0.0, 1.0],
        vec![0.0f32, 1.0, 0.0, 1.0],
    ]; // Vectors to compare against

    // Compute cosine similarity using both GPU and CPU
    let gpu_similarities = compute_cosine_similarity(&target, &vectors);
    let cpu_similarities = compute_cosine_similarity_cpu(&target, &vectors);

    // Print results from both implementations
    println!("\nGPU Results:");
    for (i, sim) in gpu_similarities.iter().enumerate() {
        println!("Similarity with vector {}: {}", i, sim);
    }

    println!("\nCPU Results:");
    for (i, sim) in cpu_similarities.iter().enumerate() {
        println!("Similarity with vector {}: {}", i, sim);
    }
}

fn compute_cosine_similarity_cpu(target: &[f32], vectors: &[Vec<f32>]) -> Vec<f32> {
    let mut similarities = Vec::with_capacity(vectors.len());

    for vector in vectors {
        // Verify vector length matches target
        assert_eq!(vector.len(), target.len(), "All vectors must have the same length");

        let mut dot_product = 0.0;
        let mut target_norm = 0.0;
        let mut vector_norm = 0.0;

        // Calculate dot product and norms
        for i in 0..target.len() {
            dot_product += target[i] * vector[i];
            target_norm += target[i] * target[i];
            vector_norm += vector[i] * vector[i];
        }

        // Calculate final norms
        target_norm = target_norm.sqrt();
        vector_norm = vector_norm.sqrt();

        // Calculate cosine similarity
        let similarity = if target_norm > 0.0 && vector_norm > 0.0 {
            dot_product / (target_norm * vector_norm)
        } else {
            0.0
        };

        similarities.push(similarity);
    }

    similarities
}

