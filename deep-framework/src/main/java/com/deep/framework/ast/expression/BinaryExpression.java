package com.deep.framework.ast.expression;

public class BinaryExpression extends Expression {
    private enum Operator {}
    private Expression left;
    private Expression right;
    private Operator operator;
}
