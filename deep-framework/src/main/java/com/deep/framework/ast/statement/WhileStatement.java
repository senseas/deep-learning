package com.deep.framework.ast.statement;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.expression.Expression;

public class WhileStatement extends Statement {
    private Expression condition;
    private Statement body;

    public WhileStatement(Node prarent) {
        super(prarent);
    }
}
