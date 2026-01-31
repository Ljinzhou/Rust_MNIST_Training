use ndarray::{Array1, Array2, Axis, s};
use rand::seq::SliceRandom;

use mnist_training::neural_network::NeuralNetwork;
use mnist_training::{load_mnist_data, one_hot_encode};

fn main() {
    let base_path = "data/MNIST/raw";
    let train_images_path = format!("{}/train-images-idx3-ubyte", base_path);
    let train_labels_path = format!("{}/train-labels-idx1-ubyte", base_path);

    // 加载训练数据
    let (x_train, y_train): (Array2<f32>, Array1<u8>) =
        load_mnist_data(&train_images_path, &train_labels_path);

    println!("X_train shape: {:?}", x_train.shape()); // [60000, 784]
    println!("Y_train shape: {:?}", y_train.shape()); // [60000]

    // 将标签转化为One-Hot编码
    let y_train_one_hot = one_hot_encode(&y_train, 10);

    print!("数据初始化完毕，共计加载{}张图片", x_train.shape()[0]);

    // 初始化模型
    let mut model = NeuralNetwork::new(784, 128, 10, 0.5);
    println!("模型初始化完毕！");

    // 训练模型
    let epochs = 100; // 训练轮数
    let batch_size = 64; // 批处理大小
    let num_samples = x_train.shape()[0]; // 样本总数
    let num_batches = num_samples / batch_size; // 批次数
    println!(
        "训练轮数epochs: {}, 批处理大小batch_size: {}, 样本总数num_samples: {}, 批次数num_batches: {}",
        epochs, batch_size, num_samples, num_batches
    );
    println!("开始训练模型...");

    use indicatif::{ProgressBar, ProgressStyle};

    let mut prev_loss = f32::MAX;

    for epoch in 1..=epochs {
        // 打乱索引
        let mut indices: Vec<usize> = (0..num_samples).collect();
        indices.shuffle(&mut rand::rng());

        // 根据打乱后的索引重排数据
        let x_shuffled = x_train.select(Axis(0), &indices);
        let y_shuffled = y_train_one_hot.select(Axis(0), &indices);

        let mut epochs_loss = 0.0;

        // 创建进度条
        let pb = ProgressBar::new(num_batches as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) 损失: {msg}")
            .expect("进度条模板错误")
            .progress_chars("#>-"));
        pb.set_message("计算中...");

        // Mini-batch 梯度下降
        for i in (0..num_samples).step_by(batch_size) {
            let end = std::cmp::min(i + batch_size, num_samples);
            let x_batch = x_shuffled.slice(s![i..end, ..]).to_owned();
            let y_batch = y_shuffled.slice(s![i..end, ..]).to_owned();

            // 前向传播
            let preds = model.forward(&x_batch);

            // 计算 Loss (仅做展示)
            let loss = model.compute_loss(&preds, &y_batch);
            epochs_loss += loss;

            // 反向传播并更新权重
            model.backward(&x_batch, &y_batch);

            // 更新进度条
            pb.inc(1);
            pb.set_message(format!("{:.4}", loss));
        }

        let avg_loss = epochs_loss / num_batches as f32;
        pb.finish_with_message(format!("第 {} 轮完成，平均损失: {:.4}", epoch, avg_loss));

        // 检查 Loss 变化是否收敛
        if (prev_loss - avg_loss).abs() < 0.001 {
            println!("在第 {} 轮训练后loss已无明显变化，训练结束！", epoch);
            break;
        }
        prev_loss = avg_loss;
    }

    model.save_model("model.json");
    println!("模型保存成功！");
}
