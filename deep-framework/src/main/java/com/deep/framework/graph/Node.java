package com.deep.framework.graph;

import java.io.Serializable;

public interface Node<N> extends Serializable {
    <M> M compute();

    void gradient();

    <M> M getInput(int i);

    <M> M getOutput();

    void setOutput(N o);
}
