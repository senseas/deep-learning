package com.deep.framework.ast.statement;

import com.deep.framework.ast.Node;

import java.util.List;

public class ConstructorStatement extends Statement {
    public String access;
    public String name;
    public List<Statement> agrments;

    public List<Statement> body;

    public ConstructorStatement(Node prarent) {
        super(prarent);
    }
}
