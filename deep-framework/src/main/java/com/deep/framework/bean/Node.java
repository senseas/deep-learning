package com.deep.framework.bean;

public interface Node<N> {
    <M> M compute();

    void gradient();

    <M> M getInput(int i);

    <M> M getOutput();

    void setOutput(N o);
}
