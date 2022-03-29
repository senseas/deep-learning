package com.deep.framework.jit.statement;

import com.deep.framework.jit.Parser;
import com.deep.framework.jit.lexer.TokenType;
import lombok.Data;

import java.util.ArrayList;
import java.util.List;

import static com.deep.framework.jit.lexer.TokenType.AT;
import static com.deep.framework.jit.lexer.TokenType.DOT;
import static javax.lang.model.SourceVersion.isIdentifier;

@Data
public class AnnotationStatement implements Statement {
    public List<Object> packages = new ArrayList<>();
    public TokenType start = AT;

    public void parser(Statement parent, Object obj, List<Object> list) {
        if (obj.equals(start)) {
            Parser.statementList.add(this);
            Object o = list.remove(0);
            while (!list.isEmpty()) {
                o = list.remove(0);
                if (o instanceof String && isIdentifier((String) o)) {
                    packages.add(o);
                    return;
                }
            }
        }
    }

}
