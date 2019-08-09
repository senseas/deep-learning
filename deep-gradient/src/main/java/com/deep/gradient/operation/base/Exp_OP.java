package com.deep.gradient.operation.base;

import com.deep.gradient.graph.Graph;
import com.deep.gradient.operation.Node;
import lombok.Data;

@Data
public class Exp_OP implements Node {

    public Exp_OP(Node... input) {
        this.input = input;
    }

    public Object compute() {
        None inx = input[0].getOutput(), iny = input[1].getOutput();
        Double valx = inx.getValue(), valy = iny.getValue();
        return Math.pow(valx, valy);
    }

    public void gradient() {
        None inx = input[0].getOutput(), iny = input[1].getOutput();
        Double valx = inx.getValue(), valy = iny.getValue();
        iny.setGrad(Math.pow(valx, valy));
    }

    public void setOutput(Object out) {
        if (out instanceof Double)
            output = new None((Double) output);
    }

    public void setGraph(Node[] input) {

    }

    private Node<None>[] input;
    private Graph graph;
    private Object output;
    private String name = "exp";
}
