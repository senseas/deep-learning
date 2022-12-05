package com.deep.framework.ast.expression;

import com.deep.framework.ast.Node;
import lombok.Data;

import java.util.Objects;
import java.util.stream.Collectors;

@Data
public class ArrayExpression extends Expression {
    public ArrayExpression(Node prarent) {
        super(prarent);
    }

    @Override
    public String toString() {
        return "[".concat(getChildrens().stream().map(Objects::toString).collect(Collectors.joining())).concat("]");
    }
}