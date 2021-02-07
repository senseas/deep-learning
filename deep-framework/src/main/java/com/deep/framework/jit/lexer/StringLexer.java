package com.deep.framework.jit.lexer;

import lombok.Data;

@Data
public class StringLexer implements Lexer {
    private String type = "string";
    private Lexer lexer;
    public transient final static String start = "\"", end = "\"";

    public StringLexer(Lexer lexer) {
        this.lexer = lexer;
    }

    public String getEnd() {
        return end;
    }
}