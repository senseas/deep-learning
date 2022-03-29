package com.deep.framework.jit.statement;

import com.deep.framework.jit.Parser;
import com.deep.framework.jit.lexer.TokenType;
import lombok.Data;

import java.util.ArrayList;
import java.util.List;

import static com.deep.framework.jit.lexer.TokenType.LBRACE;
import static com.deep.framework.jit.lexer.TokenType.RBRACE;

@Data
public class BlockStatement implements Statement {
    public List<Statement> statements = new ArrayList<>();
    public static TokenType start = LBRACE, end = RBRACE;

    public void parser(Statement parent, Object obj, List<Object> list) {
        if (obj.equals(start)) {
            parent.statements.add(this);
            StringBuilder buffer = new StringBuilder();
            while (!list.isEmpty()) {
                Object o = list.get(0);
                list.remove(0);
                if (!o.equals(end)) {
                    Parser.parser(this, RBRACE, list);
                    buffer.append(o);
                } else {
                    return;
                }
            }
        }
    }
}
