package com.deep.framework.jit.statement;

import com.deep.framework.jit.Parser;
import com.deep.framework.jit.lexer.TokenType;
import lombok.Data;

import java.util.List;

import static javax.lang.model.SourceVersion.isIdentifier;

@Data
public class AnnotationStatement implements Statement {
    public String name;
    public TokenType tokenType = TokenType.AT;

    public void parser(Statement parent, Object obj, List<Object> list) {
        if (obj.equals(TokenType.AT)) {
            Parser.statementList.add(this);
            StringBuilder buffer = new StringBuilder();
            while (!list.isEmpty()) {
                Object o = list.get(0);
                list.remove(0);
                if (o instanceof String) {
                    if (isIdentifier((String) o)) {
                        buffer.append(o);
                    }
                } else if (!o.equals(TokenType.LPAREN) || o.equals(TokenType.RPAREN)) {
                    name = buffer.toString();
                    return;
                } else {
                    buffer.append(o);
                }
            }
        }
    }

}
