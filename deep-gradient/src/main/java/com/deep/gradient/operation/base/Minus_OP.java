package com.deep.gradient.operation.base;

import com.deep.gradient.graph.Graph;
import com.deep.gradient.operation.Node;
import lombok.Data;

@Data
public class Minus_OP implements Node {

    public Minus_OP(Node... input) {
        this.input = input;
    }

    public Object compute() {
        if (input.length == 1) {
            None inx = input[0].getOutput();
            Double valx = inx.getValue();
            return -valx;
        } else {
            None inx = input[0].getOutput(), iny = input[1].getOutput();
            Double valx = inx.getValue(), valy = iny.getValue();
            return valx - valy;
        }
    }

    public void gradient() {
        if (input.length == 1) {
            None inx = input[0].getOutput(), out = (None) getOutput();
            Double gradx = -out.getGrad();
            inx.setGrad(gradx);
        } else {
            None inx = input[0].getOutput(), iny = input[1].getOutput();
            Double valx = inx.getValue(), valy = iny.getValue();
            inx.setGrad(valx);
            iny.setGrad(valy);
        }
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
    private String name = "minus";
}
