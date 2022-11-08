package com.deep.framework.ast.statement;

import com.deep.framework.ast.Node;

import java.util.List;

public class StatementList extends Statement {
    public List<StatementList> statements;

    public StatementList(Node prarent) {
        super(prarent);
    }
}
