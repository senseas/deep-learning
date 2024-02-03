package com.deep.framework.optimizer;

import com.deep.framework.graph.Tensor;

import static com.deep.framework.lang.Shape.zeros;

public class AdamOptimizer {
    private double[] m, n;
    private static double lr = 0.0003;
    public static double beta1 = 0.9, beta2 = 0.99, eps = 1e-6;

    public AdamOptimizer(int[] shape) {
        this.m = zeros(shape);
        this.n = zeros(shape);
    }

    public void adam(Tensor tensor) {
        double m1 = beta1 * m[tensor.getIdx()] + (1 - beta1) * tensor.grad();
        double n1 = beta2 * n[tensor.getIdx()] + (1 - beta1) * tensor.grad() * tensor.grad();
        double m2 = m1 / (1 - beta1);
        double n2 = n1 / (1 - beta2);
        double data = tensor.data() - lr * m2 / (Math.pow(n2, 0.5) + eps);
        m[tensor.getIdx()] = m1;
        n[tensor.getIdx()] = n1;
        tensor.data(data);
    }
}
