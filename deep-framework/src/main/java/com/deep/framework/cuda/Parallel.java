package com.deep.framework.cuda;

import com.deep.framework.graph.Tensor;

import java.io.Serializable;

public interface Parallel extends Serializable {

    void compute(Tensor tensor);

    void gradient(Tensor tensor);

}