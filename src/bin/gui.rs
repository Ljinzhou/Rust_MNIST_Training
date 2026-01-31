use eframe::egui;
use mnist_training::neural_network::NeuralNetwork;
use std::path::Path;

fn main() -> eframe::Result<()> {
    // 设置窗口选项
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 500.0]) // 初始窗口大小
            .with_resizable(true),
        ..Default::default()
    };

    // 运行应用程序
    eframe::run_native(
        "MNIST 手写数字识别",
        options,
        Box::new(|cc| {
            // 设置中文字体
            let mut fonts = egui::FontDefinitions::default();
            // 尝试加载 Windows 系统自带的中文字体 (微软雅黑)
            let font_path = "C:\\Windows\\Fonts\\msyh.ttc";

            if let Ok(data) = std::fs::read(font_path) {
                // 加载字体数据
                fonts.font_data.insert(
                    "microsoft_yahei".to_owned(),
                    std::sync::Arc::new(egui::FontData::from_owned(data)),
                );

                // 将中文字体优先级设为最高
                fonts
                    .families
                    .get_mut(&egui::FontFamily::Proportional)
                    .unwrap()
                    .insert(0, "microsoft_yahei".to_owned());
                fonts
                    .families
                    .get_mut(&egui::FontFamily::Monospace)
                    .unwrap()
                    .insert(0, "microsoft_yahei".to_owned());

                // 应用字体配置
                cc.egui_ctx.set_fonts(fonts);
            } else {
                eprintln!(
                    "警告：未在 {} 找到中文字体，中文可能会显示为乱码。",
                    font_path
                );
            }

            Ok(Box::new(MyApp::new(cc)))
        }),
    )
}

struct MyApp {
    // 神经网络模型
    model: Option<NeuralNetwork>,
    // 错误信息（如果模型加载失败）
    error_message: Option<String>,

    // 画板分辨率 (默认 28)
    resolution: usize,
    // 实际的像素数据 (0.0 - 1.0, 扁平化存储)
    grid_data: Vec<f32>,

    // 预测结果
    prediction_probs: Vec<f32>,
    predicted_digit: Option<usize>,
}

impl MyApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let model_path = "model.json";
        let (model, error) = if Path::new(model_path).exists() {
            match std::fs::File::open(model_path) {
                Ok(file) => {
                    let reader = std::io::BufReader::new(file);
                    match serde_json::from_reader(reader) {
                        Ok(m) => (Some(m), None),
                        Err(e) => (None, Some(format!("模型解析失败: {}", e))),
                    }
                }
                Err(e) => (None, Some(format!("无法打开模型文件: {}", e))),
            }
        } else {
            (
                None,
                Some("未找到 model.json，请先运行训练程序！".to_string()),
            )
        };

        // 初始化画板
        let resolution = 28;
        let grid_data = vec![0.0; resolution * resolution];

        Self {
            model,
            error_message: error,
            resolution,
            grid_data,
            prediction_probs: vec![0.0; 10], // 0-9 的概率
            predicted_digit: None,
        }
    }

    // 清空画板
    fn clear_grid(&mut self) {
        self.grid_data.fill(0.0);
        self.prediction_probs.fill(0.0);
        self.predicted_digit = None;
    }

    // 当分辨率改变时重置
    fn resize_grid(&mut self, new_res: usize) {
        if new_res < 28 {
            return;
        }
        self.resolution = new_res;
        self.grid_data = vec![0.0; new_res * new_res];
        self.clear_grid();
    }

    // 执行预测
    fn predict(&mut self) {
        // 先计算并克隆输入数据，避免持有 grid_data 的引用
        let input_vec = if self.resolution == 28 {
            self.grid_data.clone()
        } else {
            self.downsample_to_28x28()
        };

        // 然后再借用 model
        if let Some(model) = &mut self.model {
            // 2. 调用模型预测
            self.prediction_probs = model.predict_probabilities(&input_vec);
            self.predicted_digit = Some(model.predict_class(&input_vec));
        }
    }

    //简单的平均池化下采样
    fn downsample_to_28x28(&self) -> Vec<f32> {
        let mut output = vec![0.0; 28 * 28];
        let scale = self.resolution as f32 / 28.0;

        for y in 0..28 {
            for x in 0..28 {
                // 映射回原图的区域
                let start_x = (x as f32 * scale).floor() as usize;
                let start_y = (y as f32 * scale).floor() as usize;
                let end_x = ((x + 1) as f32 * scale).ceil() as usize;
                let end_y = ((y + 1) as f32 * scale).ceil() as usize;

                let mut sum = 0.0;
                let mut count = 0.0;

                for orig_y in start_y..end_y.min(self.resolution) {
                    for orig_x in start_x..end_x.min(self.resolution) {
                        sum += self.grid_data[orig_y * self.resolution + orig_x];
                        count += 1.0;
                    }
                }

                output[y * 28 + x] = if count > 0.0 { sum / count } else { 0.0 };
            }
        }
        output
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // 设置暗色主题 (黑背景)
        ctx.set_visuals(egui::Visuals::dark());

        egui::CentralPanel::default().show(ctx, |ui| {
            // 错误提示
            if let Some(err) = &self.error_message {
                ui.colored_label(egui::Color32::RED, err);
                return;
            }

            ui.horizontal(|ui| {
                // === 左边：控制与画板 ===
                ui.vertical(|ui| {
                    // 控制栏
                    ui.horizontal(|ui| {
                        ui.label("分辨率:");
                        let mut res = self.resolution;
                        if ui
                            .add(egui::DragValue::new(&mut res).range(28..=100).speed(1))
                            .changed()
                        {
                            self.resize_grid(res);
                        }

                        if ui.button("清空画板 (Reset)").clicked() {
                            self.clear_grid();
                        }
                    });

                    ui.add_space(10.0);

                    // 画板区域
                    // 我们使用 Sense::drag() 来检测鼠标拖动
                    let canvas_size = 400.0; // 画板显示大小
                    let (response, painter) =
                        ui.allocate_painter(egui::Vec2::splat(canvas_size), egui::Sense::drag());

                    // 绘制背景框
                    let rect = response.rect;
                    painter.rect(
                        rect,
                        0.0,
                        egui::Color32::BLACK,
                        egui::Stroke::new(1.0, egui::Color32::GRAY),
                        egui::StrokeKind::Middle,
                    );

                    // 处理鼠标输入 (绘画逻辑)
                    if response.hovered() || response.dragged() {
                        if let Some(pos) = response.interact_pointer_pos() {
                            // 计算鼠标在网格中的坐标
                            let x_rel = pos.x - rect.min.x;
                            let y_rel = pos.y - rect.min.y;

                            let cell_size = canvas_size / self.resolution as f32;
                            let grid_x = (x_rel / cell_size).floor() as usize;
                            let grid_y = (y_rel / cell_size).floor() as usize;

                            if grid_x < self.resolution && grid_y < self.resolution {
                                // 简单的笔刷：涂抹当前格及其周围的格子，模拟笔触粗细
                                let brush_radius = 1; // 笔刷半径
                                let value_add = 0.5; // 增加的亮度值 (多次涂抹会变白)

                                for dy in -(brush_radius as isize)..=brush_radius as isize {
                                    for dx in -(brush_radius as isize)..=brush_radius as isize {
                                        let nx = grid_x as isize + dx;
                                        let ny = grid_y as isize + dy;

                                        if nx >= 0
                                            && nx < self.resolution as isize
                                            && ny >= 0
                                            && ny < self.resolution as isize
                                        {
                                            let idx =
                                                (ny as usize) * self.resolution + (nx as usize);
                                            // 增加亮度，限制在 1.0 (白色)
                                            self.grid_data[idx] =
                                                (self.grid_data[idx] + value_add).min(1.0);
                                        }
                                    }
                                }

                                // 只要画了，就触发预测
                                self.predict();
                            }
                        }
                    }

                    // 渲染网格内容
                    let cell_size = canvas_size / self.resolution as f32;
                    for y in 0..self.resolution {
                        for x in 0..self.resolution {
                            let val = self.grid_data[y * self.resolution + x];
                            if val > 0.01 {
                                // 只画非黑的格子
                                let cell_rect = egui::Rect::from_min_size(
                                    rect.min
                                        + egui::vec2(x as f32 * cell_size, y as f32 * cell_size),
                                    egui::Vec2::splat(cell_size),
                                );
                                // 颜色越亮越白
                                let color = egui::Color32::from_gray((val * 255.0) as u8);
                                painter.rect_filled(cell_rect, 0.0, color);
                            }
                        }
                    }
                });

                ui.add_space(20.0);

                // === 右边：预测结果 ===
                ui.vertical(|ui| {
                    ui.label("实时预测结果:");
                    ui.add_space(10.0);

                    // 显示数字 0-9 及其概率
                    for i in 0..10 {
                        ui.horizontal(|ui| {
                            // 数字标签
                            ui.label(format!("{}:", i));

                            // 进度条
                            let prob = self.prediction_probs[i];
                            let bar =
                                egui::ProgressBar::new(prob).text(format!("{:.1}%", prob * 100.0));

                            // 如果是预测结果，把进度条标亮 (绿色)
                            if Some(i) == self.predicted_digit {
                                // egui 默认样式不太好改颜色，这里通过加粗文字区分
                                ui.add(bar.fill(egui::Color32::GREEN));
                            } else {
                                ui.add(bar);
                            }
                        });
                    }

                    ui.add_space(20.0);
                    // 显示最终结论
                    if let Some(digit) = self.predicted_digit {
                        ui.heading(format!("预测结果: {}", digit));

                        // 简单的置信度文本
                        let conf = self.prediction_probs[digit];
                        if conf > 0.8 {
                            ui.label("(非常确信)");
                        } else if conf > 0.5 {
                            ui.label("(有点可能)");
                        } else {
                            ui.label("(不太确定)");
                        }
                    } else {
                        ui.label("请在左侧画板写数字...");
                    }
                });
            });
        });
    }
}
