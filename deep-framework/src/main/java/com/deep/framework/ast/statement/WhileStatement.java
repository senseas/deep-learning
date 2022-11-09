package com.deep.framework.ast.statement;

import com.deep.framework.ast.expression.Expression;

public class WhileStatement extends Statement {
    private Expression condition;
    private BlockStatement body;
}
