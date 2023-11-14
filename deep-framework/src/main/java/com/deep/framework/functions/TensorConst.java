package com.deep.framework.functions;

import com.deep.framework.lang.Tenser;
import lombok.experimental.Accessors;

import java.util.stream.IntStream;

import static com.deep.framework.lang.Shape.size;

@Accessors(chain = true)
public class TensorConst extends Tensor {

    public TensorConst(String value) {
        super(value);
        this.output = new Tenser<>(this);
    }

    public TensorConst(int[] shape, String value) {
        super(shape);
        this.output = Tensors(value);
    }

    public void setGrad(TensorConst grad) {}

    public Tenser<Tensor> Tensors(String value) {
        return new Tenser<>(IntStream.range(0, size(shape)).mapToObj(i -> new TensorConst(value)).toArray(TensorConst[]::new), shape);
    }

}