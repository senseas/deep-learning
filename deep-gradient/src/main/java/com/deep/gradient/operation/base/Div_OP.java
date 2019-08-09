package com.deep.gradient.operation.base;

import com.deep.gradient.graph.Graph;
import com.deep.gradient.operation.Node;
import lombok.Data;

@Data
public class Div_OP implements Node {

    public Div_OP(Node... input) {
        this.input = input;
    }

    public Object compute() {
        None inx = input[0].getOutput(), iny = input[1].getOutput();
        Double valx = inx.getValue(), valy = iny.getValue();
        return valx / valy;
    }

    public void gradient() {
        None inx = input[0].getOutput(), iny = input[1].getOutput(), out = (None) getOutput();
        Double valx = inx.getValue(), valy = iny.getValue();
        Double grad = out.getGrad();
        inx.setGrad(grad * valy / Math.pow(valy, 2));
        iny.setGrad(-grad * valx / Math.pow(valy, 2));
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
    private String name = "div";
}
