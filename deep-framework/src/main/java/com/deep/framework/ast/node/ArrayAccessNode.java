package com.deep.framework.ast.node;

import com.deep.framework.ast.expression.Expression;

public interface ArrayAccessNode {

    Expression getExpression();

    Expression getIndex();
}