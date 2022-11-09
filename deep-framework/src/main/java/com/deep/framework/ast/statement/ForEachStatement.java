package com.deep.framework.ast.statement;

import com.deep.framework.ast.expression.Expression;

import java.util.List;

public class ForEachStatement extends Statement {
    private List<Expression> initialization;
    private Expression compare;
    private List<Expression> update;
    private BlockStatement body;
}