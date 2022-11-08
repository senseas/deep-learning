package com.deep.framework.ast.statement;

import com.deep.framework.ast.Node;

import java.util.List;

public class BlockStatement extends Statement {
    public List<Statement> list;

    public BlockStatement(Node prarent) {
        super(prarent);
    }
}