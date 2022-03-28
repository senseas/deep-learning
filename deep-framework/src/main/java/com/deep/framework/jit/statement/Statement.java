package com.deep.framework.jit.statement;

import com.deep.framework.jit.lexer.TokenType;

import java.util.ArrayList;
import java.util.List;

public interface Statement {
    List<Statement> statements = new ArrayList<>();

    default void parser(Statement parent, Object o, List<Object> list) { }

    default TokenType getTokenType(){
        return null;
    };
}
