import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 第一部分：数据生成
def generate_delivery_data(n_samples=1000):
    """
    生成模拟配送数据
    真实关系: time = 8 + 3*distance + 噪声
    """
    np.random.seed(42)
    distances = np.random.uniform(1, 20, n_samples)  # 距离1-20公里
    noise = np.random.normal(0, 2, n_samples)        # 噪声
    times = 8 + 3 * distances + noise                # 基础时间8分钟 + 每公里3分钟
    
    return distances, times

# 生成数据
distances, times = generate_delivery_data()

# 数据预处理
# 将数据转换为PyTorch张量
X = torch.FloatTensor(distances).reshape(-1, 1)
y = torch.FloatTensor(times).reshape(-1, 1)

# 数据标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = torch.FloatTensor(scaler_X.fit_transform(X))
y_scaled = torch.FloatTensor(scaler_y.fit_transform(y))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

print(f"训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# 第二部分：模型定义
class DeliveryTimePredictor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1):
        super().__init__()
        # 定义网络层
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# 第三部分：训练循环
def train_model(model, X_train, y_train, X_val=None, y_val=None, epochs=1000, learning_rate=0.01):
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 记录训练过程中的损失
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # 训练模式
        model.train()
        
        # 前向传播
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # 验证损失
        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                val_losses.append(val_loss.item())
        
        # 每100个epoch打印一次损失
        if (epoch + 1) % 100 == 0:
            if val_losses:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_losses[-1]:.4f}')
            else:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}')
    
    return train_losses, val_losses

# 第四部分：模型评估
def evaluate_model(model, X_test, y_test, scaler_X, scaler_y):
    # 模型预测
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test)
    
    # 将预测结果转换回原始尺度
    y_pred = scaler_y.inverse_transform(y_pred_scaled.numpy())
    y_true = scaler_y.inverse_transform(y_test.numpy())
    
    # 计算评估指标
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    
    print(f"评估指标:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # 可视化结果
    plt.figure(figsize=(15, 5))
    
    # 子图1: 预测 vs 真实值
    plt.subplot(1, 3, 1)
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('真实配送时间')
    plt.ylabel('预测配送时间')
    plt.title('预测 vs 真实值')
    plt.grid(True, alpha=0.3)
    
    # 子图2: 残差图
    plt.subplot(1, 3, 2)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title('残差图')
    plt.grid(True, alpha=0.3)
    
    # 子图3: 原始数据与拟合曲线
    plt.subplot(1, 3, 3)
    # 生成一些测试距离用于绘制拟合曲线
    test_distances = np.linspace(1, 20, 100).reshape(-1, 1)
    test_distances_scaled = torch.FloatTensor(scaler_X.transform(test_distances))
    
    with torch.no_grad():
        predicted_times_scaled = model(test_distances_scaled)
    
    predicted_times = scaler_y.inverse_transform(predicted_times_scaled.numpy())
    
    plt.scatter(scaler_X.inverse_transform(X_test.numpy()), y_true, alpha=0.6, label='测试数据')
    plt.plot(test_distances, predicted_times, 'r-', lw=2, label='模型预测')
    plt.plot(test_distances, 8 + 3 * test_distances, 'g--', lw=2, label='真实关系')
    plt.xlabel('距离 (公里)')
    plt.ylabel('配送时间 (分钟)')
    plt.title('配送时间预测')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return y_pred, y_true, {'MSE': mse, 'MAE': mae, 'RMSE': rmse}

# 主程序
if __name__ == "__main__":
    # 执行完整流程
    
    # 1. 创建模型
    model = DeliveryTimePredictor()
    print("模型结构:")
    print(model)
    
    # 2. 训练模型
    print("\n开始训练...")
    train_losses, val_losses = train_model(
        model, X_train, y_train, 
        X_val=X_test, y_val=y_test,  # 使用测试集作为验证集
        epochs=1000, 
        learning_rate=0.01
    )
    
    # 3. 绘制训练损失
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='训练损失')
    if val_losses:
        plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练过程')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 4. 评估模型
    print("\n模型评估:")
    y_pred, y_true, metrics = evaluate_model(model, X_test, y_test, scaler_X, scaler_y)
    
    # 5. 打印模型学到的参数（与真实关系比较）
    print(f"\n真实关系: time = 8 + 3 * distance")
    print("模型学到的近似关系可以通过权重分析...")
    
    # 6. 示例预测
    print("\n示例预测:")
    test_distance = 10  # 10公里
    test_distance_scaled = torch.FloatTensor(scaler_X.transform([[test_distance]]))
    
    with torch.no_grad():
        predicted_time_scaled = model(test_distance_scaled)
        predicted_time = scaler_y.inverse_transform(predicted_time_scaled.numpy())
    
    true_time = 8 + 3 * test_distance
    print(f"距离: {test_distance} 公里")
    print(f"模型预测时间: {predicted_time[0][0]:.2f} 分钟")
    print(f"真实关系计算时间: {true_time:.2f} 分钟")
    print(f"误差: {abs(predicted_time[0][0] - true_time):.2f} 分钟")