package com.deep.framework.jit.statement;

import lombok.Data;

import java.util.List;

@Data
public class ClassStatement implements Statement {
    public String access;
    public String name;
    public List<ConstructorStatement> constructor;
    public List<ExpressionStatement> expression;
    public List<FunctionStatement> function;
}
