package com.deep.framework.jit.statement;

import com.deep.framework.jit.lexer.Lexer;
import lombok.Data;

import java.util.ArrayList;
import java.util.List;

@Data
public class FunctionStatement implements Statement {
    public String name;
    public String returnValue;
    public List<String> modifier;
    public Lexer argument;
    public Lexer body;

    public FunctionStatement() {
        this.modifier = new ArrayList<>();
    }
}
