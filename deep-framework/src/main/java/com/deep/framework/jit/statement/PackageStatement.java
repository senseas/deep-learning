package com.deep.framework.jit.statement;

import com.deep.framework.jit.Parser;
import com.deep.framework.jit.lexer.TokenType;
import lombok.Data;

import java.util.ArrayList;
import java.util.List;

import static com.deep.framework.jit.lexer.TokenType.*;

@Data
public class PackageStatement implements Statement {
    public List<Object> packages = new ArrayList<>();
    public TokenType start = PACKAGE, end = SEMI;

    public void parser(Statement parent, Object obj, List<Object> list) {
        if (obj.equals(start)) {
            Parser.statementList.add(this);
            while (!list.isEmpty()) {
                Object o = list.remove(0);
                if (o.equals(end)) {
                    return;
                } else {
                    packages.add(o);
                }
            }
        }
    }
}
