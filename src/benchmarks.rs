extern crate bencher;
use bencher::Bencher;

#[bench]
fn bench_threaded_multiply(b: &mut Bencher) {
    b.iter(|| {
        let a = Matrix::new(200, 200, &|row, col| { row + col });
        let b = Matrix::new(200, 200, &|row, col| { row + col });

        bencher::black_box(a * b);
    });
}

#[bench]
fn bench_single_thread_multiply(b: &mut Bencher) {
    b.iter(|| {
        let a = Matrix::new(200, 200, &|row, col| { row + col });
        let b = Matrix::new(200, 200, &|row, col| { row + col });

        bencher::black_box(a.slow_mul(&b));
    });
}

#[bench]
fn bench_entrywise_product_test(b: &mut Bencher) {
    b.iter(|| {
        let a = Matrix::new(1000, 1000, &|row, col| { row + col });
        let b = Matrix::new(1000, 1000, &|row, col| { row - col });
        bencher::black_box(a.entrywise_product(&b));
    });
}

#[bench]
fn bench_agent_training(b: &mut Bencher) {
    b.iter(|| {
        let game_description = GameDescription::FourInARow;
        let twoplayerscore = &TwoPlayerScore;

        let num_agents = 4;
        let mut agents: Vec<UnsafeCell<Mutex<NeuralNetworkAgent>>> = Vec::with_capacity(num_agents);
        let layers = vec![
            LayerDescription {
                num_neurons: 50_usize,
                function_descriptor: ActivationFunctionDescriptor::TwoPlayerScore
            },
            LayerDescription {
                num_neurons: 1_usize,
                function_descriptor: ActivationFunctionDescriptor::TwoPlayerScore
            }
        ];
        for i in 0..num_agents {
            agents.push(UnsafeCell::new(Mutex::new(construct_agent(game_description, &layers))));
        }

        let trainer = KSuccessionTrainer::new(game_description);
        bencher::black_box(battle_agents(1, &trainer, &agents));
    });
}

#[bench]
fn matrix_operations_cpu(b: &mut Bencher) {
    b.iter(|| {
        let A = Matrix::new(1000, 1000, &|row, col| (row + 2 * col) as f32);
        let B = Matrix::new(1000, 1000, &|row, col| (2 * row + col) as f32);

        bencher::black_box(&(&A + &B).entrywise_product(&A) * &A);
    });
}

#[bench]
fn matrix_operations_gpu(b: &mut Bencher) {
    b.iter(|| {
        let A = Matrix::new(1000, 1000, &|row, col| (row + 2 * col) as f32);
        let B = Matrix::new(1000, 1000, &|row, col| (2 * row + col) as f32);

        let handle_a = MatrixHandle::from_matrix(&A);
        let handle_b = MatrixHandle::from_matrix(&B);

        bencher::black_box(&(&handle_a + &handle_b).entrywise_product(&handle_a) * &handle_a);
    });
}