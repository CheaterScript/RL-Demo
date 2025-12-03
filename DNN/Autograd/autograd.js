class Variable {
  constructor(value, name = '') {
    this.value = value;
    this.grad = 0;
    this.name = name;
    this._backward = () => {};
    this._prev = new Set();
  }

  // 加法操作
  add(other) {
    const otherVar = other instanceof Variable ? other : new Variable(other);
    const out = new Variable(this.value + otherVar.value);
    out._prev = new Set([this, otherVar]);
    
    out._backward = () => {
      this.grad += out.grad;
      otherVar.grad += out.grad;
    };
    
    return out;
  }

  // 乘法操作
  mul(other) {
    const otherVar = other instanceof Variable ? other : new Variable(other);
    const out = new Variable(this.value * otherVar.value);
    out._prev = new Set([this, otherVar]);
    
    out._backward = () => {
      this.grad += otherVar.value * out.grad;
      otherVar.grad += this.value * out.grad;
    };
    
    return out;
  }

  // 减法操作
  sub(other) {
    const otherVar = other instanceof Variable ? other : new Variable(other);
    return this.add(otherVar.neg());
  }

  // 取负操作
  neg() {
    const out = new Variable(-this.value);
    out._prev = new Set([this]);
    
    out._backward = () => {
      this.grad += -out.grad;
    };
    
    return out;
  }

  // 除法操作
  div(other) {
    const otherVar = other instanceof Variable ? other : new Variable(other);
    const out = new Variable(this.value / otherVar.value);
    out._prev = new Set([this, otherVar]);
    
    out._backward = () => {
      this.grad += (1 / otherVar.value) * out.grad;
      otherVar.grad += (-this.value / (otherVar.value * otherVar.value)) * out.grad;
    };
    
    return out;
  }

  // 幂运算
  pow(power) {
    const out = new Variable(Math.pow(this.value, power));
    out._prev = new Set([this]);
    
    out._backward = () => {
      this.grad += power * Math.pow(this.value, power - 1) * out.grad;
    };
    
    return out;
  }

  // 指数运算
  exp() {
    const out = new Variable(Math.exp(this.value));
    out._prev = new Set([this]);
    
    out._backward = () => {
      this.grad += out.value * out.grad;
    };
    
    return out;
  }

  // 对数运算
  log() {
    const out = new Variable(Math.log(this.value));
    out._prev = new Set([this]);
    
    out._backward = () => {
      this.grad += (1 / this.value) * out.grad;
    };
    
    return out;
  }

  // ReLU激活函数
  relu() {
    const out = new Variable(Math.max(0, this.value));
    out._prev = new Set([this]);
    
    out._backward = () => {
      this.grad += (this.value > 0 ? 1 : 0) * out.grad;
    };
    
    return out;
  }

  // Sigmoid激活函数
  sigmoid() {
    const sig = 1 / (1 + Math.exp(-this.value));
    const out = new Variable(sig);
    out._prev = new Set([this]);
    
    out._backward = () => {
      this.grad += sig * (1 - sig) * out.grad;
    };
    
    return out;
  }

  // Tanh激活函数
  tanh() {
    const t = Math.tanh(this.value);
    const out = new Variable(t);
    out._prev = new Set([this]);
    
    out._backward = () => {
      this.grad += (1 - t * t) * out.grad;
    };
    
    return out;
  }

  // 反向传播计算梯度
  backward() {
    // 拓扑排序计算所有节点
    const topo = [];
    const visited = new Set();
    
    const buildTopo = (v) => {
      if (!visited.has(v)) {
        visited.add(v);
        v._prev.forEach(buildTopo);
        topo.push(v);
      }
    };
    
    buildTopo(this);
    
    // 设置输出梯度为1
    this.grad = 1;
    
    // 反向传播
    for (let i = topo.length - 1; i >= 0; i--) {
      topo[i]._backward();
    }
  }

  // 清除梯度
  zeroGrad() {
    const clearGrad = (v) => {
      v.grad = 0;
      v._prev.forEach(clearGrad);
    };
    clearGrad(this);
  }

  // 链式操作支持
  toString() {
    return `Variable(value=${this.value}, grad=${this.grad})`;
  }
}

// 辅助函数，支持运算符重载
function v(value, name = '') {
  return new Variable(value, name);
}

// 向量操作扩展
class Tensor {
  constructor(data, requiresGrad = true) {
    this.data = data.map(val => new Variable(val, requiresGrad));
    this.shape = [data.length];
  }

  // 向量加法
  add(other) {
    if (!(other instanceof Tensor)) {
      other = new Tensor(Array(this.data.length).fill(other), false);
    }
    
    const result = new Tensor([], false);
    result.data = this.data.map((v, i) => v.add(other.data[i]));
    result.shape = [...this.shape];
    return result;
  }

  // 向量点积
  dot(other) {
    let result = this.data[0].mul(other.data[0]);
    for (let i = 1; i < this.data.length; i++) {
      result = result.add(this.data[i].mul(other.data[i]));
    }
    return result;
  }

  // 应用函数到每个元素
  map(fn) {
    const result = new Tensor([], false);
    result.data = this.data.map(fn);
    result.shape = [...this.shape];
    return result;
  }

  // 求和
  sum() {
    let result = this.data[0];
    for (let i = 1; i < this.data.length; i++) {
      result = result.add(this.data[i]);
    }
    return result;
  }

  // 计算梯度
  backward() {
    const sum = this.sum();
    sum.backward();
  }

  // 获取梯度
  grad() {
    return this.data.map(v => v.grad);
  }

  // 获取值
  value() {
    return this.data.map(v => v.value);
  }
}

// 损失函数
class Loss {
  // 均方误差
  static mse(predictions, targets) {
    let loss = v(0);
    for (let i = 0; i < predictions.length; i++) {
      const diff = predictions[i].sub(targets[i]);
      loss = loss.add(diff.pow(2));
    }
    return loss.div(predictions.length);
  }

  // 交叉熵损失
  static crossEntropy(predictions, targets) {
    let loss = v(0);
    for (let i = 0; i < predictions.length; i++) {
      const logProb = predictions[i].log();
      loss = loss.add(targets[i].mul(logProb).neg());
    }
    return loss.div(predictions.length);
  }
}

// 优化器基类
class Optimizer {
  constructor(parameters, lr = 0.01) {
    this.parameters = parameters;
    this.lr = lr;
  }

  step() {
    throw new Error('step() must be implemented');
  }

  zeroGrad() {
    this.parameters.forEach(param => {
      if (Array.isArray(param)) {
        param.forEach(p => p.zeroGrad && p.zeroGrad());
      } else {
        param.zeroGrad && param.zeroGrad();
      }
    });
  }
}

// 随机梯度下降
class SGD extends Optimizer {
  step() {
    this.parameters.forEach(param => {
      if (Array.isArray(param)) {
        param.forEach(p => {
          if (p instanceof Variable) {
            p.value -= this.lr * p.grad;
          }
        });
      } else if (param instanceof Variable) {
        param.value -= this.lr * param.grad;
      }
    });
  }
}

// Adam优化器
class Adam extends Optimizer {
  constructor(parameters, lr = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) {
    super(parameters, lr);
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.epsilon = epsilon;
    this.m = new Map();
    this.v = new Map();
    this.t = 0;
  }

  step() {
    this.t += 1;
    
    this.parameters.forEach(param => {
      if (param instanceof Variable) {
        if (!this.m.has(param)) {
          this.m.set(param, 0);
          this.v.set(param, 0);
        }

        const m = this.beta1 * this.m.get(param) + (1 - this.beta1) * param.grad;
        const v = this.beta2 * this.v.get(param) + (1 - this.beta2) * param.grad * param.grad;
        
        const mHat = m / (1 - Math.pow(this.beta1, this.t));
        const vHat = v / (1 - Math.pow(this.beta2, this.t));
        
        param.value -= this.lr * mHat / (Math.sqrt(vHat) + this.epsilon);
        
        this.m.set(param, m);
        this.v.set(param, v);
      }
    });
  }
}

// 神经网络层
class Linear {
  constructor(inputSize, outputSize) {
    this.weights = Array.from({ length: outputSize }, () => 
      Array.from({ length: inputSize }, () => v(Math.random() * 0.1))
    );
    this.bias = Array.from({ length: outputSize }, () => v(Math.random() * 0.1));
  }

  forward(x) {
    const outputs = [];
    for (let i = 0; i < this.weights.length; i++) {
      let sum = v(0);
      for (let j = 0; j < this.weights[i].length; j++) {
        sum = sum.add(this.weights[i][j].mul(x[j]));
      }
      outputs.push(sum.add(this.bias[i]));
    }
    return outputs;
  }

  parameters() {
    return [...this.weights.flat(), ...this.bias];
  }
}

// 使用示例
function example() {
  console.log('=== 基础使用示例 ===');
  
  // 创建变量
  const x = v(2, 'x');
  const y = v(3, 'y');
  
  // 构建计算图: f(x, y) = x^2 + y^2 + x*y
  const z = x.pow(2).add(y.pow(2)).add(x.mul(y));
  
  console.log(`f(${x.value}, ${y.value}) = ${z.value}`);
  
  // 计算梯度
  z.backward();
  console.log(`∂f/∂x = ${x.grad}`);
  console.log(`∂f/∂y = ${y.grad}`);
  
  console.log('\n=== 神经网络示例 ===');
  
  // 简单的线性回归
  const X = [v(1), v(2), v(3), v(4)];
  const Y = [v(2), v(4), v(6), v(8)];
  
  // 参数
  const w = v(0.5, 'w');
  const b = v(0, 'b');
  
  const optimizer = new SGD([w, b], 0.01);
  
  // 训练循环
  for (let epoch = 0; epoch < 100; epoch++) {
    optimizer.zeroGrad();
    
    let totalLoss = v(0);
    for (let i = 0; i < X.length; i++) {
      const prediction = w.mul(X[i]).add(b);
      const loss = prediction.sub(Y[i]).pow(2);
      totalLoss = totalLoss.add(loss);
    }
    
    totalLoss = totalLoss.div(X.length);
    totalLoss.backward();
    optimizer.step();
    
    if (epoch % 20 === 0) {
      console.log(`Epoch ${epoch}: w=${w.value.toFixed(4)}, b=${b.value.toFixed(4)}, loss=${totalLoss.value.toFixed(4)}`);
    }
  }
  
  console.log('\n训练结果:');
  console.log(`w = ${w.value.toFixed(4)}, b = ${b.value.toFixed(4)}`);
  
  // 测试预测
  const testX = v(5);
  const prediction = w.mul(testX).add(b);
  console.log(`预测 f(5) = ${prediction.value.toFixed(4)}`);
  
  console.log('\n=== 激活函数示例 ===');
  const a = v(0.5, 'a');
  const sig = a.sigmoid();
  const relu = a.relu();
  const tanh = a.tanh();
  
  sig.backward();
  console.log(`Sigmoid(${a.value}) = ${sig.value.toFixed(4)}, grad = ${a.grad.toFixed(4)}`);
}

// 运行示例
  example();

//  sigmoid'(x) = d(x)z(1-z)