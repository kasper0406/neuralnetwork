use rand::thread_rng;
use rand::seq::SliceRandom;
use neuralnetwork::{ ActivationFunctionDescriptor, NeuralNetwork, LayerDescription };
use neuralnetwork::simple::SimpleNeuralNetwork;
use neuralnetwork::{ DropoutType, Regulizer };
use simplematrixhandle::SimpleMatrixHandle;
use metalmatrixhandle::MetalMatrixHandle;
use matrixhandle::MatrixHandle;
use matrix::Matrix;
use std::fs::File;
use std::io::Read;
use std::iter::FromIterator;

const image_size: usize = 16 * 16;

// type MatrixHandleType = SimpleMatrixHandle;
type MatrixHandleType = MetalMatrixHandle;
type Network = SimpleNeuralNetwork<MatrixHandleType>;

#[derive(Clone)]
struct ImageSample {
    values: Matrix<f32>,
    label: Matrix<f32>
}

fn load_kasper_samples() -> Vec<ImageSample> {
    let mut result = vec![];
    for category in &vec![("handwritten", 3), ("machine", 4)] {
        for i in 0..10 {
            let filename = format!("./data/digits/{}_{}.raw", i, category.0);
            let mut file = File::open(&filename).expect("File not found");

            let mut pixels = Vec::with_capacity(image_size);
            let mut pixel_buffer = [0; 3];
            while let Ok(read_bytes) = file.read(&mut pixel_buffer) {
                if read_bytes == 0 {
                    break;
                }
                pixels.push(1_f32 - ((pixel_buffer[0] as f32 + pixel_buffer[1] as f32 + pixel_buffer[2] as f32) / (3 * 255) as f32));

                // Monster hack to adjust for image format
                if category.1 == 4 {
                    let mut tmp_buffer = [0; 1];
                    file.read(&mut tmp_buffer);
                }
            }

            result.push(ImageSample {
                label: Matrix::new(10, 1, &|row, _col| if i == row { 1_f32 } else { 0_f32 }),
                values: Matrix::new(image_size, 1, &|row, _col| pixels[row])
            });
        }
    }

    return result;
}

fn load_samples() -> Vec<ImageSample> {
    let mut file = File::open("./data/semeion.data").expect("Data file not found!");
    let mut content = String::new();
    file.read_to_string(&mut content).expect("Failed reading dataset file!");
    return content.trim().split("\n")
        .map(|sample| {
            let raw_values: Vec<f32> = sample.trim().split(" ").map(|value| {
                return value.parse().unwrap();
            }).collect();

            return ImageSample {
                values: Matrix::new(image_size, 1, &|row, _col| raw_values[row]),
                label: Matrix::new(10, 1, &|row, _col| raw_values[image_size + row])
            }
        })
        .collect();
}

fn print_sample(sample: Matrix<f32>) {
    for i in 0..16 {
        let mut values = Vec::with_capacity(16);
        for j in 0..16 {
            values.push(sample[(i * 16 + j, 0)].to_string());
        }
        println!("{}", values.join(""));
    }
}

fn construct_network() -> Network {
    let layers = vec![
        /*
        LayerDescription {
            num_neurons: 80_usize,
            function: relu
        }, */
        /*
        LayerDescription {
            num_neurons: 50_usize,
            function_descriptor: ActivationFunctionDescriptor::Sigmoid
        },*/
        LayerDescription {
            num_neurons: 25_usize,
            function_descriptor: ActivationFunctionDescriptor::Sigmoid
        },
        LayerDescription {
            num_neurons: 10_usize,
            function_descriptor: ActivationFunctionDescriptor::Sigmoid
        }
    ];

    let mut nn = Network::new(image_size, layers.clone());
    nn.set_dropout(DropoutType::Weight(0.10));
    // nn.set_dropout(DropoutType::Neuron(0.05));
    nn.set_regulizer(Some(Regulizer::WeightPeanalizer(0.00003_f32)));

    return nn;
}

fn samples_to_handle(samples: &[&ImageSample]) -> (MatrixHandleType, MatrixHandleType) {
    let values = MatrixHandleType::from_matrix(Matrix::new(image_size, samples.len(),
        &|row, col| samples[col].values[(0, row)]));

    let labels = MatrixHandleType::from_matrix(Matrix::new(10, samples.len(),
        &|row, col| samples[col].label[(0, row)]));

    return (values, labels);
}

fn compute_avg_error(network: &mut Network, samples: &[ImageSample]) -> f32 {
    let mut samples_ref = Vec::with_capacity(samples.len());
    samples.into_iter().for_each(|sample| samples_ref.push(sample));
    let (values, labels) = samples_to_handle(samples_ref.as_slice());
    return network.error(&values, &labels);
}

pub fn test_digit_classification() {
    let mut rng = thread_rng();

    let mut samples = load_samples();
    let mut network = construct_network();

    samples.shuffle(&mut rng);
    let training_samples = &samples[0..1000];
    let test_samples = &samples[1000..];

    let mut kasper_samples = load_kasper_samples().shuffle(&mut rng);

    let alpha = 0.02_f32;
    let beta = 0.95_f32;
    const samples_per_batch: usize = 200;

    for round in 0..1000 {
        let in_sample_error = compute_avg_error(&mut network, training_samples);
        println!("Avg error after {} rounds: {} in-sample, {} out-of-sample",
            round, in_sample_error, compute_avg_error(&mut network, test_samples));

        let mut momentums = None;
        for _ in 0..50 {
            // Train 100 samples at the same time
            let mut selected_samples = Vec::with_capacity(samples_per_batch);
            training_samples.choose_multiple(&mut rng, samples_per_batch).for_each(|sample| selected_samples.push(sample));
            let (values, labels) = samples_to_handle(selected_samples.as_slice());
            
            momentums = Some(network.train(&values, &labels, alpha, beta, &momentums));
        }
    }

    println!("Avg error after training: {} in-sample, {} out-of-sample",
            compute_avg_error(&mut network, training_samples),
            compute_avg_error(&mut network, test_samples));

    println!("");
    println!("Classification matrix - rows are labels, columns are predictions:");
    let mut classification_matrix = Matrix::new(10, 10, &|row, col| 0);
    let mut total_misclassified = 0;

    // TODO:This can be combined into one prediction instead
    for sample in test_samples {
        let prediction_vector = MatrixHandleType::to_matrix(&network.predict(&MatrixHandleType::from_matrix(sample.values.clone())));

        let mut prediction = 0;
        let mut actual = 0;
        for i in 0..10 {
            if prediction_vector[(i, 0)] > prediction_vector[(prediction, 0)] {
                prediction = i;
            }
            if sample.label[(i, 0)] > sample.label[(actual, 0)] {
                actual = i;
            }
        }

        classification_matrix[(actual, prediction)] += 1;
        if actual != prediction {
            total_misclassified += 1;
        }
    }
    println!("{}", classification_matrix);
    println!("Misclassified {} out of {} ({}%)", total_misclassified, test_samples.len(),
        (total_misclassified as f64 / test_samples.len() as f64) * 100_f64);
}
