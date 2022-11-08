package com.deep.framework.ast.statement;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.expression.Expression;

import java.util.List;

public class ForStatement extends Statement {
    private List<Expression> initialization;
    private Expression compare;
    private List<Expression> update;
    private Statement body;

    public ForStatement(Node prarent) {
        super(prarent);
    }
}
