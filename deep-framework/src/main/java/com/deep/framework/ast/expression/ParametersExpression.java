package com.deep.framework.ast.expression;

import com.deep.framework.ast.Node;

import java.util.Objects;
import java.util.stream.Collectors;

public class ParametersExpression extends Expression {
    public ParametersExpression(Node prarent) {
        super(prarent);
    }

    public String toString() {
        if (Objects.nonNull(getTokenType())) return "(".concat(getTokenType().toString()).concat(")");
        if (getChildrens().isEmpty()) return "()";
        return "(".concat(getChildrens().stream().map(Objects::toString).collect(Collectors.joining(" "))).concat(")");
    }
}