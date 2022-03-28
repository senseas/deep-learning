package com.deep.framework.jit.statement;

import com.deep.framework.jit.Parser;
import com.deep.framework.jit.lexer.TokenType;
import lombok.Data;

import java.util.List;

import static com.deep.framework.jit.lexer.TokenType.RBRACE;

@Data
public class BlockStatement implements Statement {
    public String name;
    public static TokenType tokenType = TokenType.LBRACE;
    public void parser(Statement parent, Object obj, List<Object> list) {
        if (obj.equals(TokenType.LBRACE)) {
            parent.statements.add(this);
            StringBuilder buffer = new StringBuilder();
            while (!list.isEmpty()) {
                Object o = list.get(0);
                list.remove(0);
                if (o.equals(RBRACE)) {
                    name = buffer.toString();
                    return;
                } else {
                    Parser.parser(this, RBRACE, list);
                    buffer.append(o);
                }
            }
        }
    }
}
