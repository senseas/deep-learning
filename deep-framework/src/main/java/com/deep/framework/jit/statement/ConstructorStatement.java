package com.deep.framework.jit.statement;

import java.util.List;

public class ConstructorStatement implements Statement {
    public String access;
    public String name;
    public List<Statement> agrments;

    public List<FunctionStatement> body;
}
