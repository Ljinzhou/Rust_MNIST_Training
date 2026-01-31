pub mod neural_network;

use ndarray::{Array1, Array2};
use std::fs::File;
use std::io::Read;

pub fn load_mnist_data(images_path: &str, labels_path: &str) -> (Array2<f32>, Array1<u8>) {
    // 读取图像
    let mut f_img = File::open(images_path).expect("无法打开图像文件");

    // 读取16字节 头信息
    let mut img_header = [0u8; 16];
    f_img
        .read_exact(&mut img_header)
        .expect("无法读取图像头信息");

    // 解析头信息
    let num_imgs = u32::from_be_bytes(img_header[4..8].try_into().unwrap()) as usize;
    let rows = u32::from_be_bytes(img_header[8..12].try_into().unwrap()) as usize;
    let cols = u32::from_be_bytes(img_header[12..16].try_into().unwrap()) as usize;
    let num_pixels = rows * cols;

    println!(
        "num_imgs: {}, rows: {}, cols: {}, num_pixels: {}",
        num_imgs, rows, cols, num_pixels
    );

    // 读取所有像素字节
    let mut img_buf = Vec::new();
    f_img.read_to_end(&mut img_buf).expect("无法读取图像");

    // 转化u8 -> f32归一化(/255.0)
    let img_data: Vec<f32> = img_buf.into_iter().map(|x| x as f32 / 255.0).collect();

    // 构建ndarray (Array2)
    // 形状 [N, 784]
    let images = Array2::from_shape_vec((num_imgs, num_pixels), img_data)
        .expect("构建图像矩阵失败：数据大小与形状不匹配");

    // 读取标签
    let mut f_lbl = File::open(labels_path).expect("无法打开标签文件");

    // 读取8字节 头信息
    let mut lbl_header = [0u8; 8];
    f_lbl
        .read_exact(&mut lbl_header)
        .expect("读取标签头信息失败");

    let num_lbls = u32::from_be_bytes(lbl_header[4..8].try_into().unwrap()) as usize;

    println!("num_lbls: {}", num_lbls);

    // 读取所有标签字节
    let mut lbl_buf = Vec::new();
    f_lbl.read_to_end(&mut lbl_buf).expect("无法读取标签");

    // 构建ndarray (Array1)
    // 形状 [N]
    let labels = Array1::from_vec(lbl_buf);

    (images, labels)
}

// 将标签转换为 One-Hot 编码
// 输入: labels [N] (Array1<u8>)
// 输出: one_hot [N, 10] (Array2<f32>)
pub fn one_hot_encode(labels: &Array1<u8>, num_classes: usize) -> Array2<f32> {
    // 初始化矩阵
    // 使用 ndarray::Array2::zeros 创建一个二维矩阵。
    // 形状 (shape) 为：(样本数量, 10)。
    // 初始值全为 0.0
    let mut one_hot = Array2::zeros((labels.len(), num_classes)); // [N, 10]
    // 置热点 (Hot bit)
    // one_hot[[行, 列]] = 1.0
    // 行 (i): 第 i 个样本。
    // 列 (label): 如果标签是 5，就将第 5 列设为 1.0，其余列保持 0.0。
    for (i, &label) in labels.iter().enumerate() {
        one_hot[[i, label as usize]] = 1.0;
    }
    one_hot
}
