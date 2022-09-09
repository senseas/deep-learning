package com.deep.framework.jit.statement;

import com.deep.framework.jit.lexer.TokenType;
import lombok.Data;

import java.util.ArrayList;
import java.util.List;

import static com.deep.framework.jit.lexer.TokenType.*;
import static javax.lang.model.SourceVersion.isIdentifier;

@Data
public class ClassStatement implements Statement {
    public List<Object> implement = new ArrayList<>();
    public List<FunctionStatement> extend = new ArrayList<>();
    public List<Object> generics = new ArrayList<>();
    public BlockStatement body;

    public String access;
    public String name;
    public TokenType start = CLASS, end = RBRACE;

    public void parser(Statement parent, Object obj, List<Object> lexers) {
        if (obj.equals(CLASS)) {
            parent.statements.add(this);
            StringBuilder buffer = new StringBuilder();
            Object o = lexers.remove(0);
            while (true) {
                o = lexers.remove(0);
                if (o instanceof String && isIdentifier((String) o)) {
                    name = (String) o;
                } else if (o.equals(LT)) {

                } else if (o.equals(GT)) {

                } else if (o.equals(IMPLEMENTS)) {
                    implement(implement, lexers);
                } else if (o.equals(LBRACE)) {
                    body = new BlockStatement();
                    body.parser(parent, o, lexers);
                    return;
                } else if (o.equals(RBRACE)) {
                    name = buffer.toString();
                    body = new BlockStatement();
                    body.parser(parent, o, lexers);
                    return;
                } else {
                    buffer.append(o);
                }
            }
        }
    }

    private void implement(List<Object> implement, List<Object> lexers) {
        while (true) {
            Object o = lexers.remove(0);
            if (o instanceof String && isIdentifier((String) o)) {
                implement.add(o);
            } else if (o.equals(LBRACE)) {
                lexers.add(0, o);
                return;
            }
        }
    }
}
