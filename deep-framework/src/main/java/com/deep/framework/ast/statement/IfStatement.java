package com.deep.framework.ast.statement;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.expression.Expression;

public class IfStatement extends Statement {
    private Expression condition;
    private Statement thenStatement;
    private Statement elseStatement;
    private BlockStatement body;

    public IfStatement(Node prarent) {
        super(prarent);
    }
}
