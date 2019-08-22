package com.deep.framework.bean;

import java.io.Serializable;

public interface Node<N> extends Serializable {
    <M> M compute();

    void gradient();

    <M> M getInput(int i);

    <M> M getOutput();

    void setOutput(N o);
}
