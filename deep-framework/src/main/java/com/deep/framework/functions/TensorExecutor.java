package com.deep.framework.functions;

import lombok.Data;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import static com.deep.framework.lang.ForEach.forBack;
import static com.deep.framework.lang.ForEach.forEach;

@Data
public class TensorExecutor implements Serializable {
    private Tensor tensor;
    private Tensor[] operators;
    private Tensor[] params;

    public List<Tensor> operator = new LinkedList<>();
    public List<Tensor> param = new LinkedList<>();

    public TensorExecutor(Tensor tensor) {
        this.tensor = tensor;
        deepEach(tensor);
        operators = operator.toArray(Tensor[]::new);
        params = param.toArray(Tensor[]::new);
    }

    public void run() {
        forward();
        backward();
    }

    public void forward() {
        forEach(operators.length, i -> operators[i].forward());
    }

    public void backward() {
        tensor.getOutput().forEach((Tensor none, int i) -> none.setGrad(new Tensor("g" + tensor.getId() + "[" + i + "]")));
        forBack(operators.length, i -> operators[i].backward());
        merge(tensor);
    }

    private void deepEach(Tensor tensor) {
        if (operator.contains(tensor)) return;

        if (tensor instanceof TensorFunction) {
            for (Tensor o : tensor.getInput()) {
                deepEach(o);
            }
            tensor.getFunction().forEach(this::deepEach);
            operator.add(tensor);
        } else if (tensor instanceof TensorOperator) {
            for (Tensor o : tensor.getInput()) {
                deepEach(o);
            }
            operator.add(tensor);
        } else if (!(tensor instanceof TensorConst) && !param.contains(tensor)) {
            param.add(tensor);
        }
    }

    private void merge(Tensor tensor) {
        for (Tensor o : tensor.getInput()) {
            o.getOutput().forEach((out) -> {
                Tensor grad = out.getGrad();
                List<String> list = new ArrayList<>();
                while (true) {
                    grad.reducer();
                    if (list.contains(grad.getData())) return;
                    list.add(grad.getData());
                }
            });
        }
    }

}