package com.deep.gradient.operation;


import com.deep.gradient.graph.Graph;

public interface Node<N> {
    N compute();

    void gradient();

    N getOutput();

    void setOutput(N out);

    Graph getGraph();

    void setGraph(Node... input);
}
