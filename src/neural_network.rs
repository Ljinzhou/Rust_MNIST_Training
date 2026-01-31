use ndarray::{Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};

#[derive(Serialize, Deserialize)]
pub struct NeuralNetwork {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub lr: f32,

    // 权重和偏置
    pub w1: Array2<f32>,
    pub b1: Array2<f32>,
    pub w2: Array2<f32>,
    pub b2: Array2<f32>,

    // 反向传播所需的缓存
    z1: Option<Array2<f32>>,
    a1: Option<Array2<f32>>,
    z2: Option<Array2<f32>>,
    a2: Option<Array2<f32>>,
}

impl NeuralNetwork {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        learning_rate: f32,
    ) -> Self {
        let w1 = Array2::random((input_size, hidden_size), StandardNormal) * 0.01;
        let b1 = Array2::zeros((1, hidden_size));
        let w2 = Array2::random((hidden_size, output_size), StandardNormal) * 0.01;
        let b2 = Array2::zeros((1, output_size));

        NeuralNetwork {
            input_size,
            hidden_size,
            output_size,
            lr: learning_rate,
            w1,
            b1,
            w2,
            b2,
            z1: None,
            a1: None,
            z2: None,
            a2: None,
        }
    }

    // Sigmoid 激活函数
    fn sigmoid(x: &Array2<f32>) -> Array2<f32> {
        // 截断数值防止 exp 计算溢出 [-500, 500]
        let clipped = x.mapv(|v| v.max(-500.0).min(500.0));
        clipped.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    }

    // Sigmoid 激活函数的导数
    fn sigmoid_derivative(x: &Array2<f32>) -> Array2<f32> {
        let sig = Self::sigmoid(x);
        &sig * &(1.0 - &sig)
    }

    // Softmax 激活函数
    fn softmax(x: &Array2<f32>) -> Array2<f32> {
        // x shape: [batch, 10]
        // 为了数值稳定性，减去每一行的最大值

        let max_vals = x.fold_axis(Axis(1), f32::NEG_INFINITY, |&a, &b| a.max(b));
        // max_vals 是 1D 数组 [batch]，插入轴使其变为 [batch, 1] 以便广播
        let max_vals = max_vals.insert_axis(Axis(1));

        let shifted = x - &max_vals;
        let exp_x = shifted.mapv(f32::exp);

        // 计算和，并保持维度 [batch, 1]
        let sum_exp = exp_x.sum_axis(Axis(1)).insert_axis(Axis(1));

        &exp_x / &sum_exp
    }

    // 前向传播
    pub fn forward(&mut self, x: &Array2<f32>) -> Array2<f32> {
        // 第一层 (隐藏层)
        // Z1 = X dot W1 + b1
        // 注意: b1 是 [1, hidden]，ndarray 会自动广播加到每一行
        let z1 = x.dot(&self.w1) + &self.b1;
        let a1 = Self::sigmoid(&z1);

        // 第二层 (输出层)
        let z2 = a1.dot(&self.w2) + &self.b2;
        let a2 = Self::softmax(&z2);

        // 缓存中间结果用于反向传播
        self.z1 = Some(z1);
        self.a1 = Some(a1);
        self.z2 = Some(z2);
        self.a2 = Some(a2.clone());

        a2
    }
    // 反向传播
    pub fn backward(&mut self, x: &Array2<f32>, y_true_encoded: &Array2<f32>) {
        let m = x.nrows() as f32;

        // 获取缓存 (使用引用或unwrap)
        let a2 = self.a2.as_ref().unwrap();
        let a1 = self.a1.as_ref().unwrap();
        let z1 = self.z1.as_ref().unwrap();

        // 1. 计算输出层的梯度
        // dZ2 = A2 - Y (Softmax + CrossEntropy 的导数简化形式)
        let dz2 = a2 - y_true_encoded;

        // 2. 计算 W2, b2 的梯度
        // dW2 = A1.T dot dZ2 / m
        let dw2 = a1.t().dot(&dz2) / m;
        // db2 = sum(dZ2, axis=0) / m
        let db2 = dz2.sum_axis(Axis(0)).insert_axis(Axis(0)) / m;

        // 3. 计算隐藏层的梯度
        // dA1 = dZ2 dot W2.T
        let da1 = dz2.dot(&self.w2.t());
        // dZ1 = dA1 * Sigmoid'(Z1)
        let dz1 = &da1 * &Self::sigmoid_derivative(z1);

        // 4. 计算 W1, b1 的梯度
        let dw1 = x.t().dot(&dz1) / m;
        let db1 = dz1.sum_axis(Axis(0)).insert_axis(Axis(0)) / m;

        // 5. 更新参数 (梯度下降)
        self.w1 = &self.w1 - &(dw1 * self.lr);
        self.b1 = &self.b1 - &(db1 * self.lr);
        self.w2 = &self.w2 - &(dw2 * self.lr);
        self.b2 = &self.b2 - &(db2 * self.lr);
    }

    // 计算损失
    pub fn compute_loss(&self, y_pred: &Array2<f32>, y_true_encoded: &Array2<f32>) -> f32 {
        let m = y_pred.nrows() as f32;
        // 交叉熵损失: -sum(y_true * log(y_pred)) / m
        // 加上 epsilon 防止 log(0)
        let epsilon = 1e-9;
        let log_probs = y_pred.mapv(|v| (v + epsilon).ln());

        // 逐元素相乘 -> 求和 -> 除以 m -> 取反
        let loss = -(y_true_encoded * &log_probs).sum() / m;
        loss
    }

    // 保存模型 (序列化为 JSON)
    pub fn save_model(&self, path: &str) {
        let file = File::create(path).expect("无法创建模型文件");
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, self).expect("写入模型 JSON 失败");
        println!("模型已保存至: {}", path);
    }

    // 读取模型 (从 JSON 反序列化)
    #[allow(dead_code)]
    pub fn load_model(path: &str) -> Self {
        let file = File::open(path).expect("无法打开模型文件");
        let reader = BufReader::new(file);
        let model: NeuralNetwork = serde_json::from_reader(reader).expect("解析模型 JSON 失败");
        println!("模型已从: {} 加载", path);
        model
    }

    /// 接受一个扁平化的输入向量 (Vec<f32>)，返回每个类别的概率 (Vec<f32>)
    pub fn predict_probabilities(&mut self, input: &Vec<f32>) -> Vec<f32> {
        // 将 Vec<f32> 转换为 Array2<f32> [1, 784]
        let x = ndarray::Array2::from_shape_vec((1, self.input_size), input.clone())
            .expect("输入数据形状错误");

        // 前向传播
        let output = self.forward(&x);

        // 转换为 Vec<f32> 返回
        output.into_raw_vec_and_offset().0
    }

    /// 接受一个扁平化的输入向量，返回概率最大的类别索引
    pub fn predict_class(&mut self, input: &Vec<f32>) -> usize {
        let probs = self.predict_probabilities(input);

        // 找到最大值的索引
        probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap()
    }
}
