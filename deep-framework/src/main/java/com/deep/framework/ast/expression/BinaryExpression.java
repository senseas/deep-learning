package com.deep.framework.ast.expression;

import com.deep.framework.ast.Node;

public class BinaryExpression extends Expression {
    private enum Operator {}
    private Expression left;
    private Expression right;
    private Operator operator;

    public BinaryExpression(Node prarent) {
        super(prarent);
    }
}
