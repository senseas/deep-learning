package com.deep.framework.bean;

public interface Node<N> {
    <M> M compute();

    void gradient();

    <M> M getInput(int i);

    void setOutput(N o);
}
