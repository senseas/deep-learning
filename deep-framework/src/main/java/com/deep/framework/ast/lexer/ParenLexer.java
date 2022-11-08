package com.deep.framework.ast.lexer;

import lombok.Data;

@Data
public class ParenLexer implements Lexer {
    private String type = "paren";
    private transient Lexer lexer;
    public transient final static String start = "(", end = ")";

    public ParenLexer(Lexer lexer) {
        this.lexer = lexer;

    }

    public String getEnd() {
        return end;
    }

}