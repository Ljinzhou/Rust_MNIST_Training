use mnist_training::load_mnist_data; // 使用库函数
use mnist_training::neural_network::NeuralNetwork;
use ndarray::{Array1, Array2};

fn main() {
    println!("正在加载模型和测试数据...");

    // 1. 加载模型
    let model_path = "model.json";
    // 检查模型文件是否存在
    if !std::path::Path::new(model_path).exists() {
        eprintln!("错误：未找到 model.json！请先运行训练程序。");
        return;
    }

    let mut model = NeuralNetwork::load_model(model_path);
    println!("模型加载成功。");

    // 2. 加载测试数据
    let base_path = "data/MNIST/raw";
    let test_images_path = format!("{}/t10k-images-idx3-ubyte", base_path);
    let test_labels_path = format!("{}/t10k-labels-idx1-ubyte", base_path);

    let (x_test, y_test): (Array2<f32>, Array1<u8>) =
        load_mnist_data(&test_images_path, &test_labels_path);

    println!(
        "测试数据加载完毕: X 形状 {:?}, Y 形状 {:?}",
        x_test.shape(),
        y_test.shape()
    );

    // 3. One-hot 编码标签 (如果需要计算损失则需要此步骤)
    // 实际上为了计算准确率，我们只需要最大概率的索引
    // let y_test_one_hot = one_hot_encode(&y_test, 10);

    // 4. 推理
    println!("正在对 {} 个测试样本进行推理...", x_test.nrows());
    let output = model.forward(&x_test);

    // 5. 计算准确率
    let mut correct_count = 0;
    let num_samples = x_test.nrows();

    for i in 0..num_samples {
        // 获取预测类别（输出行中最大值的索引）
        // output row shape: [10]
        let prediction = output
            .row(i)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap();

        // 检查是否匹配标签
        if prediction == y_test[i] as usize {
            correct_count += 1;
        }
    }

    let accuracy = correct_count as f32 / num_samples as f32;
    println!(
        "测试准确率: {:.2}% ({}/{})",
        accuracy * 100.0,
        correct_count,
        num_samples
    );
}
