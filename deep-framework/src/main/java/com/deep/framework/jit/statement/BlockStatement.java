package com.deep.framework.jit.statement;

import lombok.Data;

import java.util.List;

@Data
public class BlockStatement implements Statement {
    public String name;
    public List<Statement> argument;
}
